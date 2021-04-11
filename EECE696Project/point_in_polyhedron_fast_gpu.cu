#if 1
//Misc Libs
#include <stdlib.h>
#include <stdio.h>
#include<math.h>
#include <string>
#include<time.h>

// Point In Polynomial Helpers
#include "pip_helpers.cuh"

//Cuda Libs
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "GpuTimer.h"

// Grid Builder 
#include "grid_builder.cuh"

// Thrust Reduction
#include <thrust\reduce.h>

#define NUM_POINTS 1000000


__device__ uint d_inside_points_count = 0;


//Well launch threads equal to grid size
// dimGrid(x*y*z//BlockSize, 1, 1)
__global__ void compute_grid_ends(uint *grid, uint *grid_end_out, uint grid_size, uint *triangle_list) {

	int global_idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (global_idx >= grid_size) return;

	uint triangle_list_idx = grid[global_idx];

	if ((triangle_list_idx >> CELL_STATE_OFFSET) == 3) {
		//cell is empty 
		//clear the empty state from both grid and grid_end cells and set thier value to 0
		//in the subsequent code we will able to figure if a cell if empty if both start and end cells point to 0
		grid[global_idx] = 0;
		grid_end_out[global_idx] = 0;
		return;
	}

	// Loop through triangles list until a termniate list flag is found
	uint terminate = 0;
	int increment = -1;
	do
	{
		increment++;
		terminate = triangle_list[triangle_list_idx + increment] & TERMINATE_SUBLIST;

	} while (terminate != TERMINATE_SUBLIST);
	grid_end_out[global_idx] = triangle_list_idx + increment;
}

// Well launch threads equal to grids y*z size 
// dimGrid(1, y//BlockSize, z//BlockSize)
__global__ void preprocess_grid(float3 *mesh, uint *grid, uint *grid_end, uint3 grid_res, float3 min, float3 cwidth, uint *triangle_list) {

	uint thread_y = blockIdx.y * blockDim.y + threadIdx.y;
	uint thread_z = blockIdx.z * blockDim.z + threadIdx.z;

	if (thread_y >= grid_res.y || thread_z >= grid_res.z) return;
	//if(thread_y != 0 || thread_z != 0) return;

	// The intial point of the segment that will cross the cell
	// It is the center point of the previous cell 
	float3 initial_point;

	// The current cell id
	int cell_grid_idx;

	// The set of the triangles that overlaps current and previous cell in each iteration
	uint* triangles = NULL;
	uint triangles_size;

	// The segment-cell intersection value
	float cell_intersection_value;

	// The singularity state of the cell
	bool is_singular;

	// The segments end point (the center point of the current cell in ech iteration) intersects a primitive (triangle) in the polyhedron object
	bool center_point_intersect_object;

	// Initilize cell intersection value
	cell_intersection_value = 0;

	// Previous Cell State // required when both cells are empty
	uint previous_cell_state = CELL_STATE_OUTSIDE;

	// Loop through entire x_axis direction cells
	for (int i = 0; i < grid_res.x; i++) {

		// Compute the segments intial point initially it is out of the grid
		initial_point = { min.x - cwidth.x / 2 + i*cwidth.x, min.y + thread_y*cwidth.y + cwidth.y / 2, min.z + thread_z*cwidth.z + cwidth.z / 2 };


		// Compute current and previous cell ids
		cell_grid_idx = thread_z * (grid_res.x*grid_res.y) + thread_y * (grid_res.x) + i;
		//printf("\nthread(1, %u, %u) visited cell %u", thread_y, thread_z, cell_grid_idx);


		// If this is the first cell accessed by the thread
		// then previous will be -1
		if (i == 0) {
			//previous_cell_grid_idx = - 1; // We set previous id to -1 to be handled correctly by get_triangles_overlap_2_cell
			get_triangles_overlap_cells(&grid[cell_grid_idx], &grid_end[cell_grid_idx], 1, triangle_list, &triangles, &triangles_size);

		}
		else { // Not the first cell
			//previous_cell_grid_idx = cell_grid_idx - 1; 
			get_triangles_overlap_cells(&grid[cell_grid_idx - 1], &grid_end[cell_grid_idx - 1], 2, triangle_list, &triangles, &triangles_size);
		}

		// Get the triangles overlapping with the two cells
		//get_triangles_overlap_2_cell(cell_grid_idx, previous_cell_grid_idx, grid, grid_end, triangle_list, &triangles, &triangles_size);

		//for (uint i = 0; i < triangles_size; i++) {// Temp
			//printf("\ncells %d %d: have %u", cell_grid_idx, cell_grid_idx - 1, triangles[i]); //Temp
		//} //Temp

		// If both cells are empty skip triangles visiting and take the state of the previous state
		if (triangles_size == 0) {
			//printf("cell %u have 0 triangles", cell_grid_idx);

			grid[cell_grid_idx] |= previous_cell_state;

			// Free triangles and reset triangles_size
			delete triangles;
			triangles = NULL;
			triangles_size = 0;

			continue;
		}

		// Intilize
		is_singular = false; // singular cell is the cell that have its center point on the polyhedron ( on a triangle edge, vertix, or plane) 
		center_point_intersect_object = false;

		// Otherwise loop on the overlapping triangles and compute intersection value and singularity
		for (int i = 0; i < triangles_size; i++) {
			cell_intersection_value += segment_triangle_intersection(mesh + 3 * triangles[i], initial_point, cwidth.x, &center_point_intersect_object);;
			is_singular |= center_point_intersect_object;
			//printf("\nThread(1,%d,%d) at cell %d #Tri %d Tri %u itr %d I.sum %f S %s", thread_y, thread_z, cell_grid_idx, triangles_size, triangles[i], i, cell_intersection_value, is_singular ? "true" : "false");
		}

		// based on the state of the previous cell and the value of the intersection
		// update the current cell state and then set the previous state variable to the 
		// computeted state of the current cell
		previous_cell_state = compute_globally_cell_state(grid, cell_grid_idx, cell_intersection_value, is_singular);

		// Set the new state as the previous state for the next iteration


		// Free triangles and reset triangles_size
		delete triangles;
		triangles = NULL;
		triangles_size = 0;

		// go to the next cell
	}

}

// Well launch threads equal to num_points
// dimGrid(num_points//blocksize, 1, 1)
__global__ void points_in_polyhedron(float3 *points, uint num_points, char *points_out, float3 *mesh, uint *grid, uint *grid_end, uint3 grid_res, float3 min, float3 max, float3 cwidth, uint *triangle_list) {

	int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
	
	if (global_idx >= num_points) return;
	//if (global_idx != 8) return;

	//atomicAdd(&d_total_thread_launched_pip, 1);

	// The point to work on
	float3 point = points[global_idx];

	// If point outside bounding box then points_out(global_idx) = 0 and return
	if ((point.x<min.x || point.x>max.x) || (point.y<min.y || point.y>max.y) || (point.z<min.z || point.z>max.z)) {
		//printf("\n\nPoint(%.2f, %.2f, %.2f) is out of the polyhedron bounding box", point.x, point.y, point.z);
		points_out[global_idx] = 0;
		return;
	}

	// Get the points's containing-cell
	uint3 cell = { floor((point.x-min.x) / cwidth.x), floor((point.y-min.y) / cwidth.y), floor((point.z-min.z) / cwidth.z) };

	// Get the containing-cell linearized id
	uint cell_id = cell.x + cell.y*grid_res.x + cell.z*grid_res.x*d_grid_res.y;

	uint cell_state;
	uint new_visited_cells;
	uint direction_jump;
	int direction = 1;

	// Keep on searching (< max threashold) in positive direction on each axis until a nonsingular cell is found
	for (new_visited_cells = 0; new_visited_cells < MAX_SEARCH_UN_SINGULAR; new_visited_cells++) {

		//printf("\nSearching for a non singular cell in %s direction", (direction==1)?"+ve":"-ve");
		// Move new_visited_cells steps in x direction
		direction_jump = 1;
		if (cell.x + direction*new_visited_cells < grid_res.x) {
			cell_state = grid[cell_id + direction*direction_jump*new_visited_cells] & CELL_STATE;
			//printf("\nVisitied Cell %d and it was %s", cell_id + direction*direction_jump*new_visited_cells, (cell_state == CELL_STATE_SINGULAR) ? "singular" : "non-singular");
			if (cell_state != CELL_STATE_SINGULAR) break;
		}

		if (new_visited_cells == 0)continue; // The below cells will be the same as the above if new_visited cells is zero

		// Move new_visited_cells steps in y direction
		direction_jump = grid_res.x;
		if (cell.y + direction*new_visited_cells < grid_res.y) {
			cell_state = grid[cell_id + direction*direction_jump*new_visited_cells] & CELL_STATE;
			//printf("\nVisitied Cell %d and it was %s", cell_id + direction*direction_jump*new_visited_cells, (cell_state == CELL_STATE_SINGULAR) ? "singular" : "non-singular");
			if (cell_state != CELL_STATE_SINGULAR) break;
		}

		// Move new_visited_cells steps in z direction
		direction_jump = grid_res.x*grid_res.y;
		if (cell.z + direction*new_visited_cells < grid_res.z) {
			cell_state = grid[cell_id + direction*direction_jump*new_visited_cells] & CELL_STATE;
			//printf("\nVisitied Cell %d and it was %s", cell_id + direction*direction_jump*new_visited_cells, (cell_state == CELL_STATE_SINGULAR) ? "singular" : "non-singular");
			if (cell_state != CELL_STATE_SINGULAR) break;
		}
	}

	if (cell_state == CELL_STATE_SINGULAR) {//There was no non-singular cell in the MAX_SEARCH_UN_SINGULAR positive ball was found

		// try in th negative ball
		direction = -1;
		//printf("\nSearching for a non singular cell in %s direction", (direction == 1) ? "+ve" : "-ve");

		// Keep on searching (< max threashold) in negative direction on each axis until a nonsingular cell is found - start from 1 as for zero it will be the same as above
		for (new_visited_cells = 1; new_visited_cells < MAX_SEARCH_UN_SINGULAR; new_visited_cells++) {

			// Move new_visited_cells steps in x direction
			direction_jump = 1;
			if ((int)(cell.x + direction*new_visited_cells) >= 0) { //i.e. >= 0
				cell_state = grid[cell_id + direction*direction_jump*new_visited_cells] & CELL_STATE;
				//printf("\nVisitied Cell %d and it was %s", cell_id + direction*direction_jump*new_visited_cells, (cell_state==CELL_STATE_SINGULAR)?"singular":"non-singular");
				if (cell_state != CELL_STATE_SINGULAR) break;
			}

			// Move new_visited_cells steps in y direction
			direction_jump = grid_res.x;
			if ((int)(cell.y + direction*new_visited_cells) >= 0) {
				cell_state = grid[cell_id + direction*direction_jump*new_visited_cells] & CELL_STATE;
				//printf("\nVisitied Cell %d and it was %s", cell_id + direction*direction_jump*new_visited_cells, (cell_state == CELL_STATE_SINGULAR) ? "singular" : "non-singular");
				if (cell_state != CELL_STATE_SINGULAR) break;
			}
			 
			// Move new_visited_cells steps in z direction
			direction_jump = grid_res.x*grid_res.y;
			if ((int)(cell.z + direction*new_visited_cells )>= 0) {
				cell_state = grid[cell_id + direction*direction_jump*new_visited_cells] & CELL_STATE;
				//printf("\nVisitied Cell %d and it was %s", cell_id + direction*direction_jump*new_visited_cells, (cell_state == CELL_STATE_SINGULAR) ? "singular" : "non-singular");
				if (cell_state != CELL_STATE_SINGULAR) break;
			}
		}

	}

	if (cell_state == CELL_STATE_SINGULAR) {//There was no non-singular cell in the MAX_SEARCH_UN_SINGULAR sized positive ball was found
		printf("\n\nError: Couldn't find a non singular cell in the radius of %d from cell %d", MAX_SEARCH_UN_SINGULAR, cell_id);
		return;
	}


	// Get the unsingular cell grid id

	uint un_singular_cell_id = cell_id + direction*direction_jump*new_visited_cells;

	//printf("\n\nThe found non-singular cell for cell %d containing point(%.2f, %.2f, %.2f) is %d\nThe process visited %d new cells using direction_jump = %d the %s direction",
		//cell_id, point.x , point.y, point.z , un_singular_cell_id, new_visited_cells, direction_jump, (direction == 1) ? "+ve" : "-ve");

	// Get the unsingular cell grid coordinates

	uint3 unsingular_cell = { un_singular_cell_id % grid_res.x, (un_singular_cell_id / grid_res.x) % grid_res.y, ((un_singular_cell_id / grid_res.x) / grid_res.y) };

	//printf("\n\nThe unsingular cell grid coordinates are (%d, %d, %d)", unsingular_cell.x, unsingular_cell.y, unsingular_cell.z);

	// Get the unsingular midpoint 
	float3 initial_point = { min.x + unsingular_cell.x*cwidth.x + cwidth.x / 2, min.y + unsingular_cell.y*cwidth.y + cwidth.y / 2, min.z + unsingular_cell.z*cwidth.z + cwidth.z / 2 };

	// Get all the grid start and end values for each cell between starting cell and unsingular cell
	uint* cell_start_ids = (uint*)malloc(sizeof(uint)*(new_visited_cells + 1)); // plus one for the initial cell
	uint* cell_end_ids = (uint*)malloc(sizeof(uint)*(new_visited_cells + 1));


	if (!cell_start_ids || !cell_end_ids) {
		printf("\nError(points_in_polyhedron): no enough dynamic memory in the heap to be allocated\n");
		return;
	}

	for (uint i = 0; i <= new_visited_cells; i++) {
		cell_start_ids[i] = grid[cell_id + direction * direction_jump * i];
		cell_end_ids[i] = grid_end[cell_id + direction * direction_jump * i];
	}

	// Get the triangles overlapping with cell
	uint* triangles = NULL;
	uint triangles_size;

	get_triangles_overlap_cells(cell_start_ids, cell_end_ids, new_visited_cells + 1, triangle_list, &triangles, &triangles_size);

	//printf("\n\nPoint(%.2f, %.2f, %.2f) segment intersected with triangles: ", point.x , point.y, point.z);
	//for (int i = 0; i < triangles_size; i++) {
	//	printf(" %d", triangles[i]);
	//}

	free(cell_start_ids);
	free(cell_end_ids);

	// Intilize
	float cell_intersection_value = 0;

	// loop on the overlapping triangles and compute intersection value 
	for (int i = 0; i < triangles_size; i++) {
		cell_intersection_value += un_algined_segment_triangle_intersection(mesh + 3 * triangles[i], initial_point, point);
		//printf("\nTriangle %d intersection value is: %f", triangles[i], cell_intersection_value);
	}

	//printf("\n\nPoint(%.2f, %.2f, %.2f) resulted in an intersection value %f", point.x , point.y , point.z, cell_intersection_value);

	//In case intial cell is empty the output will be identical to it's state (cell_intersection_value is initilized to zero)
	//Otherwise the required computations will be performed and the adequate point state is find  
	uint point_state = compute_locally_point_state(cell_state, cell_intersection_value);
	//printf("\n\nPoint(%.2f, %.2f, %.2f) is %s the polyhedron", point.x , point.y , point.z , (points_out[global_idx]) ? "inside" : "outside");

	if (point_state == CELL_STATE_INSIDE) {
		points_out[global_idx]=1;
		//atomicAdd(&d_inside_points_count, 1);
	}

	else{
		if (point_state == CELL_STATE_OUTSIDE) {
			points_out[global_idx] = 0;
		}
	}



	free(triangles);
	return;
}


int main_test_simple_object() {

	int numtri = 4;
	float3 A = make_float3(1.75, 0, 1.5);
	float3 B = make_float3(5, 6, 6);
	float3 C = make_float3(7, -2, 0);
	float3 D = make_float3(0, 6, 0);

	float3 *h_mesh = (float3*)malloc(3 * numtri * sizeof(float3));

	h_mesh[0] = B; h_mesh[1] = A; h_mesh[2] = D;
	h_mesh[3] = B; h_mesh[4] = C; h_mesh[5] = A;
	h_mesh[6] = B; h_mesh[7] = D; h_mesh[8] = C;
	h_mesh[9] = D; h_mesh[10] = A; h_mesh[11] = C;


	float3* d_mesh; uint* d_grid; uint* d_triangle_list;

	float3 min = make_float3(0, -2, 0);
	float3 max = make_float3(7, 6, 6);

	uint3 grid_resolution;
	float3 cwidth;
	float grid_density = 4;

	build_uniform_grid(h_mesh, numtri, min, max, grid_density, &d_mesh, &d_grid, &grid_resolution, &cwidth, &d_triangle_list);

	uint grid_size = grid_resolution.x*grid_resolution.y*grid_resolution.z;

	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid(ceil(grid_size / (float)BLOCK_SIZE), 1, 1);

	thrust::device_ptr<uint> d_grid_end = thrust::device_malloc<uint>(grid_size);

	printf("Grid Ends Computation Started\n");
	compute_grid_ends << <dimGrid, dimBlock >> > (d_grid, thrust::raw_pointer_cast(d_grid_end), grid_size, d_triangle_list);
	cudaDeviceSynchronize();
	print_if_cuda_error(__LINE__);
	printf("\n\n");

	//uint* h_grid_0 = (uint*)malloc(sizeof(uint)*grid_size); //Temp
	//thrust::copy(thrust::device_pointer_cast(d_grid), thrust::device_pointer_cast(d_grid) + grid_size, h_grid_0); //Temp
	//printf("\n grid: "); //Temp
	//for (int i = 0; i < 8; i++) { //Temp
	//	printf("%d ", h_grid_0[i]); //Temp
	//} //Temp

	//uint* h_grid_end = (uint*)malloc(sizeof(uint)*grid_size); //Temp
	//thrust::copy(d_grid_end, d_grid_end + grid_size, h_grid_end); //Temp
	//printf("\n grid_out: "); //Temp
	//for (int i = 0; i < 8; i++) { //Temp
	//	printf("%d ", h_grid_end[i]); //Temp
	//} //Temp

	dim3 dimBlock2(1, 16, 16);
	dim3 dimGrid2(1, ceil(grid_resolution.y / 16.0), ceil(grid_resolution.z / 16.0));
	printf("Preprocessing Grid Started\n");
	preprocess_grid << <dimGrid2, dimBlock2 >> > (thrust::raw_pointer_cast(d_mesh), thrust::raw_pointer_cast(d_grid), thrust::raw_pointer_cast(d_grid_end), grid_resolution, min, cwidth, thrust::raw_pointer_cast(d_triangle_list));
	cudaDeviceSynchronize();
	print_if_cuda_error(__LINE__);
	printf("\n\n");


	uint* h_grid = (uint*)malloc(sizeof(uint)*grid_size); //Temp
	thrust::copy(thrust::device_pointer_cast(d_grid), thrust::device_pointer_cast(d_grid) + grid_size, h_grid); //Temp
	printf("\nPre processed grid: "); //Temp
	for (int i = 0; i < 8; i++) { //Temp
		printf("%#010x ", h_grid[i]); //Temp
	} //Temp
	printf("\n");

	uint numpoints = NUM_POINTS;
	float3* h_points = (float3*)malloc(sizeof(float3)*numpoints);

	thrust::device_ptr<float3> d_points = thrust::device_malloc<float3>(numpoints);
	thrust::device_ptr<char> d_points_out = thrust::device_malloc<char>(numpoints);


	h_points[0] = make_float3(2, -1, 7); // outside bb
	h_points[1] = make_float3(1.1, -1, 1); // outside p cell 0
	h_points[2] = make_float3(4, 1.5, 2); // inside p cell 1
	h_points[3] = make_float3(2, 2, 1.5); // inside p cell 2
	h_points[4] = make_float3(2.2, 4.3, 0.2); // outsied p cell 2
	h_points[5] = make_float3(2.8, 3, 2.6); // inside p cell 2
	h_points[6] = make_float3(6, 5, 0.2); // oustside p cell 3 
	h_points[7] = make_float3(5, 0.7, 4.7); // outside p cell 5
	h_points[8] = make_float3(3.9, 5, 5); //outside p cell 7
	h_points[9] = make_float3(4.4, 2.6, 4); // outside p cell 7

	thrust::copy(h_points, h_points + numpoints, d_points);

	dim3 dimBlock3(16, 1, 1);
	dim3 dimGrid3(ceil(10 / 16.0), 1, 1);
	printf("\nPoints in polyhedron test Started\n");
	points_in_polyhedron << <dimGrid3, dimBlock3 >> > (thrust::raw_pointer_cast(d_points), numpoints, thrust::raw_pointer_cast(d_points_out), thrust::raw_pointer_cast(d_mesh), thrust::raw_pointer_cast(d_grid), thrust::raw_pointer_cast(d_grid_end), grid_resolution, min, max, cwidth, thrust::raw_pointer_cast(d_triangle_list));
	cudaDeviceSynchronize();
	print_if_cuda_error(__LINE__);
	printf("\n\n");

	char* h_points_out = (char*)malloc(sizeof(char)*numpoints); //Temp
	thrust::copy(d_points_out, d_points_out + numpoints, h_points_out); //Temp
	printf("\nPoints results: "); //Temp
	for (uint i = 0; i < numpoints; i++) { printf(" %d", h_points_out[i]); } //Temp
	printf("\n"); //Temp



	return 0;
}

void main_test_2_simple_object() {

	int numtri = 4;
	float3 A = make_float3(1.75, 0, 1.5);
	float3 B = make_float3(5, 6, 6);
	float3 C = make_float3(7, -2, 0);
	float3 D = make_float3(0, 6, 0);

	float3 *h_mesh = (float3*)malloc(3 * numtri * sizeof(float3));

	h_mesh[0] = B; h_mesh[1] = A; h_mesh[2] = D;
	h_mesh[3] = B; h_mesh[4] = C; h_mesh[5] = A;
	h_mesh[6] = B; h_mesh[7] = D; h_mesh[8] = C;
	h_mesh[9] = D; h_mesh[10] = A; h_mesh[11] = C;

	float3 segment_origin = make_float3(3.5, 1.5, 1.5);
	float3 segment_end = make_float3(1.75, 0, 1.5);

	float iv1 = un_algined_segment_triangle_intersection(&h_mesh[0], segment_origin, segment_end);
	float iv2 = un_algined_segment_triangle_intersection(&h_mesh[3], segment_origin, segment_end);
	float iv3 = un_algined_segment_triangle_intersection(&h_mesh[6], segment_origin, segment_end);
	float iv4 = un_algined_segment_triangle_intersection(&h_mesh[9], segment_origin, segment_end);

	float summation = iv1 + iv2 + iv3 + iv4;
}

void main(void) {

	float3 *h_mesh = NULL;
	float3 *d_mesh = NULL;
	uint* d_grid;
	uint* d_triangle_list;

	int numv, numtri;
	float3 min;
	float3 max;

	float grid_density = 8;
	uint3 grid_resolution;
	float3 cwidth;
	GpuTimer timer;
	float total_time = 0;


	//Load obj file model as a mesh
	//TODO tranform into function
	std::vector<float3> mesh_vector;
	std::vector<float> vertices_vector;
	std::vector<int> triangles_vector;
	char *filepath = "media/dragon.obj";

	objLoader(filepath, &triangles_vector, &vertices_vector, &numv, &numtri, &min, &max);

	toMesh(triangles_vector.data(), vertices_vector.data(), numtri, &mesh_vector);

	h_mesh = mesh_vector.data();


	timer.Start();
	build_uniform_grid(h_mesh, numtri, min, max, grid_density, &d_mesh, &d_grid, &grid_resolution, &cwidth, &d_triangle_list);
	timer.Stop();
	total_time += timer.Elapsed();

//	printf("\n\tGrid Building Kernel: %f msecs.\n", timer.Elapsed());

	uint grid_size = grid_resolution.x*grid_resolution.y*grid_resolution.z;

	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid(ceil(grid_size / (float)BLOCK_SIZE), 1, 1);

	thrust::device_ptr<uint> d_grid_end = thrust::device_malloc<uint>(grid_size);

	printf("Grid Ends Computation Started\n");
	timer.Start();
	compute_grid_ends << <dimGrid, dimBlock >> > (d_grid, thrust::raw_pointer_cast(d_grid_end), grid_size, d_triangle_list);
	cudaDeviceSynchronize();
	print_if_cuda_error(__LINE__);
	timer.Stop();
	total_time += timer.Elapsed();
	//printf("\t\n Kernel Time: %f msecs.\n", timer.Elapsed());

	printf("\n\n");

	printf("Preprocessing Grid Started\n");
	dim3 dimBlock2(1, 16, 16);
	dim3 dimGrid2(1, ceil(grid_resolution.y / 16.0), ceil(grid_resolution.z / 16.0));
	timer.Start();
	preprocess_grid << <dimGrid2, dimBlock2 >> > (thrust::raw_pointer_cast(d_mesh), thrust::raw_pointer_cast(d_grid), thrust::raw_pointer_cast(d_grid_end), grid_resolution, min, cwidth, thrust::raw_pointer_cast(d_triangle_list));
	cudaDeviceSynchronize();
	print_if_cuda_error(__LINE__);
	timer.Stop();
	total_time += timer.Elapsed();
	//printf("\t\n Kernel Time: %f msecs.\n", timer.Elapsed());
	printf("\n\n");


	uint* h_grid = (uint*)malloc(sizeof(uint)*grid_size);
	thrust::copy(thrust::device_pointer_cast(d_grid), thrust::device_pointer_cast(d_grid) + grid_size, h_grid);

	// Export grid points 
	//export_grid_points(h_grid, grid_size, min, max, grid_resolution, cwidth);

	// Generate Random Points in the polyhedron bounding box
	float3 *h_points;
	std::vector<float3> h_points_vector;
	uint numpoints = NUM_POINTS;
	char* h_points_out = (char*) malloc(sizeof(char)*numpoints);

	generate_random_points(min.x, max.x, min.y, max.y, min.z, max.z, numpoints, &h_points_vector);
	h_points = h_points_vector.data();

	//printf("Generated Points\n");
	//for (uint i = 0; i < numpoints; i++)	{
	//	if (i % 3 == 0)printf("\n");
	//	printf(" %d:(%.2f, %.2f, %.2f)", i, h_points[i].x, h_points[i].y, h_points[i].z);
	//}
	//printf("\n\n");

	thrust::device_ptr<float3> d_points = thrust::device_malloc<float3>(numpoints);
	thrust::device_ptr<char> d_points_out = thrust::device_malloc<char>(numpoints);
	thrust::copy(h_points, h_points + numpoints, d_points);

	uint h_inside_points_count = 0;

	printf("Computing Points Inclusion Started\n");
	dim3 dimBlock3(512, 1, 1);
	timer.Start();
	dim3 dimGrid3(ceil(numpoints/512.0), 1, 1);
	points_in_polyhedron<<<dimGrid3, dimBlock3>>>(thrust::raw_pointer_cast(d_points), numpoints, thrust::raw_pointer_cast(d_points_out), thrust::raw_pointer_cast(d_mesh), thrust::raw_pointer_cast(d_grid), thrust::raw_pointer_cast(d_grid_end), grid_resolution, min, max, cwidth, thrust::raw_pointer_cast(d_triangle_list));
	cudaDeviceSynchronize();
	print_if_cuda_error(__LINE__);
	timer.Stop();
	total_time += timer.Elapsed();

	printf("\t\n Kernel Time: %f msecs.\n", total_time);
	
	//cudaMemcpyFromSymbol(&h_inside_points_count, d_inside_points_count, sizeof(uint));
	//printf("\tOut of %u points, %u are inside and %u are outside", numpoints, h_inside_points_count, numpoints - h_inside_points_count);

	printf("\n\n");

	//printf("Points Inclusion State\n");
	thrust::copy(d_points_out, d_points_out + numpoints, h_points_out);
	//for (uint i = 0; i < numpoints; i++) {
	//	if(i%10==0)printf("\n");
	//	printf(" %d", h_points_out[i]);
	//}
	//printf("\n\n");

	//Export Resulting Points as Obj File
	//export_test_points_as_obj_files(h_points, h_points_out, numpoints);

	cudaFree(thrust::raw_pointer_cast(d_points));
	cudaFree(d_grid);
	cudaFree(thrust::raw_pointer_cast(d_grid_end));
	cudaFree(thrust::raw_pointer_cast(d_points_out));
	cudaFree(d_mesh);
	cudaFree(d_triangle_list);

	free(h_points);
	free(h_points_out);
	free(h_mesh);
	free(h_grid);
	
	//Learned
	// texture memory
	// warp wise thread operations
	// thrust
	// opengl
	// used dynamic allocation in the kernel
	// obj and stuff

	return;
}

#endif