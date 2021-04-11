#include <stdlib.h>

#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "helper_math.h"

#include <thrust\scan.h>
#include <thrust\device_ptr.h>
#include <thrust\device_malloc.h>
#include <thrust\fill.h>

#define BLOCK_SIZE  512
/**
* Flag that is set when it is determined that a voxel truly
* overlaps a triangle. This is written along with the
* triangle ID into the associated coarse pair element.
*/
#define COARSE_PAIR_COLLISION_OFFSET    31
#define COARSE_PAIR_COLLISION ((uint)1<<COARSE_PAIR_COLLISION_OFFSET)

/**
* Written into each element of the overlap counts array (which
* is later converted into a pointer). When this flag is set,
* space needs to be allocated in the triangle list for the
* triangles that overlap this voxel.
*/
#define DO_ALLOCATION ((uint)1<<31)

/**
* Also written into each element of the overlap counts array.
* When this flag is set, an allocation is still in progress and
* no triangle IDs can be copied into the triangle list until
* this is cleared.
*/
#define REQ_ALLOCATION ((uint)1<<30)

/**
* The triangle list contains a sublist for each occupied voxel.
* This flag is set in each element that terminates a sublist.
*/
#define TERMINATE_SUBLIST ((uint)1<<31)

__device__  int d_total_overlaps_count;
__device__  int d_triangle_list_idx;
__device__ __constant__ uint3 d_grid_res; // grid resolution x,y,z
__constant__ float3 d_smin; // secene minimum x,y,z
__constant__ float3 d_cwidth; // one voxel(cell) size x,y,z

							  // Helper Methods
__device__ inline void get_vox_aabb(float3 *pts, int3 &mins, int3 &maxs) {
	mins.x = (int)((fminf(fminf(pts[0].x, pts[1].x), pts[2].x) - d_smin.x) / d_cwidth.x);
	mins.y = (int)((fminf(fminf(pts[0].y, pts[1].y), pts[2].y) - d_smin.y) / d_cwidth.y);
	mins.z = (int)((fminf(fminf(pts[0].z, pts[1].z), pts[2].z) - d_smin.z) / d_cwidth.z);

	mins.x = clamp(mins.x, (int)0, (int)d_grid_res.x - 1);
	mins.y = clamp(mins.y, (int)0, (int)d_grid_res.y - 1);
	mins.z = clamp(mins.z, (int)0, (int)d_grid_res.z - 1);

	maxs.x = (int)((fmaxf(fmaxf(pts[0].x, pts[1].x), pts[2].x) - d_smin.x) / d_cwidth.x);
	maxs.y = (int)((fmaxf(fmaxf(pts[0].y, pts[1].y), pts[2].y) - d_smin.y) / d_cwidth.y);
	maxs.z = (int)((fmaxf(fmaxf(pts[0].z, pts[1].z), pts[2].z) - d_smin.z) / d_cwidth.z);

	maxs.x = clamp(maxs.x, (int)0, (int)d_grid_res.x - 1);
	maxs.y = clamp(maxs.y, (int)0, (int)d_grid_res.y - 1);
	maxs.z = clamp(maxs.z, (int)0, (int)d_grid_res.z - 1);
}

__device__ inline void project(const float3 axis, float3 *p, float *ret_min, float *ret_max) {

	float _dot[3] = { dot(axis, p[0]),
		dot(axis, p[1]),
		dot(axis, p[2]) };

	*ret_min = *ret_max = _dot[0];

	if (_dot[1] < *ret_min) *ret_min = _dot[1];
	else if (_dot[1] > *ret_max) *ret_max = _dot[1];

	if (_dot[2] < *ret_min) *ret_min = _dot[2];
	else if (_dot[2] > *ret_max) *ret_max = _dot[2];
}

__device__ inline void bproject(float3 axis, float3 bmin, float3 bmax, float *ret_min, float *ret_max) {

	float center = dot(axis, ((bmin + bmax) * 0.5f));
	float pextent = fabs((bmax.x - bmin.x) * 0.5f * axis.x) +
		fabs((bmax.y - bmin.y) * 0.5f * axis.y) +
		fabs((bmax.z - bmin.z) * 0.5f * axis.z);

	*ret_min = center - pextent;
	*ret_max = center + pextent;
}

__device__ inline int tri_box_axis_overlap_test(float3 axis, float3 *pts, float3 *box) {
	float tmin, tmax;
	float bmin, bmax;

	project(axis, pts, &tmin, &tmax);

	bproject(axis, box[0], box[1], &bmin, &bmax);

	if (bmin - tmax > 0.0001f) return 0;
	if (tmin - bmax > 0.0001f) return 0;
	return 1;
}

__device__ inline int tri_box_overlap_test(float3 *pts, float3 *box, float3 *edges) {

	float3 axis;

	axis = cross(edges[0], make_float3(1, 0, 0));
	if (!tri_box_axis_overlap_test(axis, pts, box)) return 0;

	axis = cross(edges[0], make_float3(0, 1, 0));
	if (!tri_box_axis_overlap_test(axis, pts, box)) return 0;

	axis = cross(edges[0], make_float3(0, 0, 1));
	if (!tri_box_axis_overlap_test(axis, pts, box)) return 0;

	axis = cross(edges[1], make_float3(1, 0, 0));
	if (!tri_box_axis_overlap_test(axis, pts, box)) return 0;

	axis = cross(edges[1], make_float3(0, 1, 0));
	if (!tri_box_axis_overlap_test(axis, pts, box)) return 0;

	axis = cross(edges[1], make_float3(0, 0, 1));
	if (!tri_box_axis_overlap_test(axis, pts, box)) return 0;

	axis = cross(edges[2], make_float3(1, 0, 0));
	if (!tri_box_axis_overlap_test(axis, pts, box)) return 0;

	axis = cross(edges[2], make_float3(0, 1, 0));
	if (!tri_box_axis_overlap_test(axis, pts, box)) return 0;

	axis = cross(edges[2], make_float3(0, 0, 1));
	if (!tri_box_axis_overlap_test(axis, pts, box)) return 0;

	// triangle normal
	axis = cross(edges[0], edges[1]);
	if (!tri_box_axis_overlap_test(axis, pts, box)) return 0;

	//
	// NOTE! Normally there are three additional tests, 
	// but they are not needed here because we already
	// know that this voxel is contained within the 
	// triangle's AABB.
	//
	return 1;
}

__host__ inline uint3 compute_grid_resolution(float3 min, float3 max, float grid_density, int num_tri) {

	uint3 grid_res;

	float volume = (max.x - min.x)*(max.y - min.y)*(max.z - min.z);
	float ratio = powf((float)(grid_density*(float)num_tri / volume), 1 / 3.);

	grid_res.x = (max.x - min.x)*ratio;
	grid_res.y = (max.y - min.y)*ratio;
	grid_res.z = (max.z - min.z)*ratio;

	return grid_res;
}

__host__ inline void print_cuda_error(int line) {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("\nError On Line %d: %s\n", line - 1, cudaGetErrorString(err));
	}
}


//Will Launch Threads Equal to Triangles Initially 
__global__ void compute_coarse_pairs(float3* mesh, int *coarse_pairs_cnts, int tri_cnt) {

	//
	// Set triangle ID to the parallel task ID
	//
	int tri_id = blockIdx.x*blockDim.x + threadIdx.x;

	//
	// There are usually more tasks than triangles.
	//
	if (tri_id >= tri_cnt) return;

	//
	// Get the three vertices of this triangle.
	//
	float3 pts[3];
	pts[0] = mesh[3 * tri_id];
	pts[1] = mesh[3 * tri_id + 1];
	pts[2] = mesh[3 * tri_id + 2];

	//printf("T %d : %.2f %.2f %.2f  %.2f %.2f %.2f  %.2f %.2f %.2f\n", tri_id, pts[0].x, pts[0].y, pts[0].z, pts[1].x, pts[1].y, pts[1].z, pts[2].x, pts[2].y, pts[2].z);

	//
	// Determine the min and max voxels of the AABB
	// of this triangle.
	//
	int3 mins, maxs;
	get_vox_aabb(pts, mins, maxs);
	//printf("T %d mins: %d %d %d max: %d %d %d\n",tri_id, mins.x, mins.y, mins.z, maxs.x, maxs.y, maxs.z);

	//
	// Count the number of voxels covered by the AABB
	// and save the total.
	//
	int3 lens = maxs - mins + make_int3(1, 1, 1);
	int cnt = lens.x * lens.y * lens.z;
	coarse_pairs_cnts[tri_id] = cnt;
}

// transform coarse_pairs_cnts into an indices array using execlusive scan
// allocate coarse_pairs based on indices + carse_pairs_cnts[numtri-1]

__global__ void tag_segments(int *coarse_pairs, int *indicies, int tri_cnt)
{
	//
	// Set triangle ID to the parallel task ID
	// skip 0 because we don't need to tag
	// triangle ID 0.
	//
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if (idx == 0) return;

	//
	// There are usually more tasks than triangles.
	//
	if (idx >= tri_cnt) return;
	coarse_pairs[indicies[idx]] = 1;
}

// Transform coarse_pairs using inclusive scan

// Will Launch threads equal to coarse pairs size
__global__ void evaluate_coarse_pairs(float3 *mesh, int coarse_pair_cnt, int *indicies, int *coarse_pairs, unsigned int *overlap_cnts) {

	//
	// Set coarse pair idx to the parallel task ID
	//
	int coarse_pair_idx = blockIdx.x*blockDim.x + threadIdx.x;
	int collision = 0;

	//
	// There are likely more tasks launched than there
	// are coarse pairs. But we need to perform a reduction
	// step afterwards, so don't just return.
	//
	if (coarse_pair_idx < coarse_pair_cnt)
	{
		//
		// Load triangle ID and convert into linear voxel id
		// within AABB (converted into grid voxel later).
		//
		int tri_id = coarse_pairs[coarse_pair_idx];
		int vox = coarse_pair_idx - indicies[tri_id];

		//
		// load triangle vertices
		//
		float3 pts[3];
		pts[0] = mesh[3 * tri_id];
		pts[1] = mesh[3 * tri_id + 1];
		pts[2] = mesh[3 * tri_id + 2];

		//
		// Get AABB in grid voxel 3D coordinates.
		//
		int3 mins, maxs;
		get_vox_aabb(pts, mins, maxs);

		//
		// Now convert the voxel within the AABB into the grid
		// x, y, z components. To be clear, the voxels
		// here are enumerated only within the AABB space, so
		// voxel 0 is the minimum corner of the AABB, but the
		// components are in the global grid space.
		//
		int3 lens = maxs - mins + make_int3(1, 1, 1);
		int xx_width = lens.x;
		int yy_width = lens.y;
		int xx = vox % xx_width;
		int yy = (vox / xx_width) % yy_width;
		int zz = ((vox / xx_width) / yy_width);

		xx += mins.x;
		yy += mins.y;
		zz += mins.z;

		//
		// Now we have the voxel and triangle that needs to
		// be precisely tested. But first a little prep work.
		//

		//
		// Calculate triangle edge vectors.
		//
		float3 edges[3] = { pts[1] - pts[0],
			pts[2] - pts[1],
			pts[0] - pts[2], };

		//
		// Make world bounding box from voxel.
		//
		float3 box[2];

		// min corner
		box[0] = make_float3(
			xx*d_cwidth.x + d_smin.x,
			yy*d_cwidth.y + d_smin.y,
			zz*d_cwidth.z + d_smin.z
		);

		// max corner
		box[1] = make_float3(
			box[0].x + d_cwidth.x,
			box[0].y + d_cwidth.y,
			box[0].z + d_cwidth.z
		);

		//
		// the whole purpose of all this...
		//
		collision = tri_box_overlap_test(pts, box, edges);

		//
		// If we have a real collision, write flag into coarse pair
		// and update the grid counter.
		//
		if (collision)
		{
			// linear voxel id in grid space
			vox = xx + (yy * (d_grid_res.x)) + (zz * d_grid_res.x * d_grid_res.y);
			coarse_pairs[coarse_pair_idx] = (tri_id | COARSE_PAIR_COLLISION); // Assumed that the total number of tri is under 2,147,483,647
			atomicAdd(&overlap_cnts[vox], 1);
		}
	}

	//
	// In order to reduce contention on the global overlap counter,
	// we perform a warp wide reduction so that 32 counts are accumulated
	// into the first thread (lane 0). Then only one access to the
	// global counter is required.
	//

	// Doesn't Work on Fermi Architecture
	//collision += __shfl_down(collision, 16);
	//collision += __shfl_down(collision, 8);
	//collision += __shfl_down(collision, 4);
	//collision += __shfl_down(collision, 2);
	//collision += __shfl_down(collision, 1);

	if (coarse_pair_idx >= coarse_pair_cnt) return;
	//int lane = threadIdx.x & 0x1f;
	//if (lane == 0) {
	atomicAdd(&d_total_overlaps_count, collision);
	//}
}

// Now we have an overlap cnt array rename it grids
// Also Coarse_pairs Now contains triangle id + collision flag
// Now we generate the triangles list

// Will Launch threads equal to coarse pairs size
__global__ void extract_grid(float3* mesh, unsigned int *grids, unsigned int *triangle_list, int *coarse_pairs, int *indicies, int total_coarse_pair_cnt) {

	//
	// Set coarse pair idx to the parallel task ID.
	// However, there are likely more tasks than pairs.
	//
	unsigned int coarse_pair_idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (coarse_pair_idx >= total_coarse_pair_cnt) return;

	//
	// Read the coarse pair and if the collision bit is set, there is
	// a valid triangle-voxel overlap, which means the data needs to
	// be added to the accelerator.
	//
	int triId = coarse_pairs[coarse_pair_idx];
	int collision = triId >> COARSE_PAIR_COLLISION_OFFSET;
	if (collision == 0) return;
	triId -= COARSE_PAIR_COLLISION;

	//
	// This segment performs the same work as was done
	// in evaluate coarse pairs. Since the voxel ID was
	// not saved off, we have to repeat it.
	//
	{
		//
		// Get the voxel ID within the AABB (this is not
		// a grid space voxel, see notes below).
		//
		int vox = coarse_pair_idx - indicies[triId];

		//
		//
		//
		float3 pts[3];
		pts[0] = mesh[3 * triId];
		pts[1] = mesh[3 * triId + 1];
		pts[2] = mesh[3 * triId + 2];

		//
		// Get AABB in grid voxel 3D coordinates.
		//
		int3 mins, maxs;
		get_vox_aabb(pts, mins, maxs);

		//
		// Now convert the voxel within the AABB into the grid
		// x, y, z components. To be clear, the voxels
		// here are enumerated only within the AABB space, so
		// voxel 0 is the minimum corner of the AABB, but the
		// components are in the global grid space.
		//
		int3 lens = maxs - mins + make_int3(1, 1, 1);
		int xx_width = lens.x;
		int yy_width = lens.y;

		int xx = vox % xx_width;
		int yy = (vox / xx_width) % yy_width;
		int zz = ((vox / xx_width) / yy_width);

		xx += mins.x;
		yy += mins.y;
		zz += mins.z;

		//
		// Convert components into grid voxel and update grids pointer.
		//
		vox = xx + (yy * (d_grid_res.x)) + (zz * d_grid_res.x * d_grid_res.y);
		grids += vox;
	}

	//
	// Attempt to clear the allocation bit flag. Idx will have the old
	// value at the specified location after the atomicAnd call.
	//
	unsigned int idx = atomicAnd(grids, ~DO_ALLOCATION);
	unsigned int add = 0;

	//
	// If the old value indicates that an allocate is needed, we are
	// the first thread to clear it. So, we have to allocate space
	// in the triangle list.
	//
	if ((idx & DO_ALLOCATION) == DO_ALLOCATION)
	{
		//
		// This means that idx is actually still a counter. Use that
		// to increment the triangle list idx position to reserve
		// our space in the triangle list.
		//
		int cnt = (idx & ~(DO_ALLOCATION | REQ_ALLOCATION));
		int start = atomicAdd(&d_triangle_list_idx, cnt);

		//
		// Since atomicAdd returned the old value at the index position,
		// the old value is where the our triangle sublist will start.
		// However, this algorithm fills in the sublist backwards.
		//
		idx = start + cnt - 1;

		//
		// This update does two things: it turns the counter into
		// a reference into the triangle list and it signals to the other
		// threads that the allocation is complete (by clearing the
		// allocation-required flag).
		//
		atomicExch(grids, idx);

		//
		// Since this will be the last element in the triangle sublist,
		// set the bit flag to terminate the list (see below).
		//
		add = TERMINATE_SUBLIST;
	}

	//
	// This is true for every thread not responsible for doing an allocation.
	//
	if (add == 0)
	{
		//
		// This loop spins, waiting for the allocation to complete.
		//
		if ((idx & REQ_ALLOCATION) == REQ_ALLOCATION)
		{
			do
			{
				idx = atomicSub(grids, 0);
			} while ((idx & REQ_ALLOCATION) == REQ_ALLOCATION);
		}

		//
		// If we are here, grids contains a valid reference where we can store
		// the coarse pair triangle ID.
		//
		idx = atomicSub(grids, 1);

		// convert old value to our idx
		idx--;
	}

	//
	// Finally! Write out the triangle ID and terminate sublist flag (if set).
	//
	triangle_list[idx] = triId + add;
}

// Now we have the grids array. Each element(voxel) in the grid contains a pointer 
// to the triangle list where we can see the triangles that overlaps the voxel
// We stop reading from the triangles list when we encounter an edn list flag

void build_uniform_grid(float3* host_mesh, int tri_cnt, float3 min, float3 max, float grid_density, float3 **d_mesh_result, uint **d_grid_result, uint3 *grid_resolution, float3* cwidth,uint **d_triangle_list_result) {

	uint3 h_grid_res;
	float3 h_cwidth;


	printf("Grid Generation Started\n");

	//Compute and initilize cwidth, and grid_res
	//Initilize smin, global_overlap_counter,  and triangle_list_idx
	{
		cudaMemcpyToSymbol(d_smin, &min, sizeof(float3));

		print_cuda_error(__LINE__);

		h_grid_res = compute_grid_resolution(min, max, grid_density, tri_cnt);
		cudaMemcpyToSymbol(d_grid_res, &h_grid_res, sizeof(uint3));
		print_cuda_error(__LINE__);
		

		h_cwidth.x = (max.x - min.x) / h_grid_res.x;
		h_cwidth.y = (max.y - min.y) / h_grid_res.y;
		h_cwidth.z = (max.z - min.z) / h_grid_res.z;
		cudaMemcpyToSymbol(d_cwidth, &h_cwidth, sizeof(float3));
		print_cuda_error(__LINE__);

		int zero = 0;
		cudaMemcpyToSymbol(d_total_overlaps_count, &zero, sizeof(int));
		print_cuda_error(__LINE__);
		cudaMemcpyToSymbol(d_triangle_list_idx, &zero, sizeof(int));
		print_cuda_error(__LINE__);
	}
	


	thrust::device_ptr<int> d_coarse_pairs_cnts;
	thrust::device_ptr<int> d_indicies;
	thrust::device_ptr<int> d_coarse_pairs;
	thrust::device_ptr<uint> d_overlap_cnts;
	thrust::device_ptr<uint> d_triangle_list;
	thrust::device_ptr<float3> d_mesh;
	thrust::device_ptr<uint> d_grid;
	int total_coarse_pair_cnt;
	int total_overlaps;
	uint grid_size;

	// Phase 1: compute coarse pairs cnts and transform it into indicies
	{
		d_mesh = thrust::device_malloc<float3>(3 * tri_cnt);
		d_coarse_pairs_cnts = thrust::device_malloc<int>(tri_cnt);
		d_indicies = thrust::device_malloc<int>(tri_cnt);

		thrust::copy(host_mesh, host_mesh + (3 * tri_cnt), d_mesh);
		//for (int i = 0; i < tri_cnt; i++) {//Temp
		//	printf("%f %f %f ", host_mesh[3 * i].x, host_mesh[3*i].y, host_mesh[3 * i].z);//Temp
		//	printf("%f %f %f ", host_mesh[3 * i+1].x, host_mesh[3 * i + 1].y, host_mesh[3 * i + 1].z);//Temp
		//	printf("%f %f %f \n", host_mesh[3 * i + 2].x, host_mesh[3 * i + 2].y, host_mesh[3 * i + 2].z);//Temp
		//}//Temp
		//printf("\n");//Temp

		dim3 dimBlock(BLOCK_SIZE, 1, 1);
		dim3 dimGrid(ceil(tri_cnt / (float)BLOCK_SIZE), 1, 1);
		printf("\tLaunched compute_coarse_pairs kernel\n");
		compute_coarse_pairs << <dimGrid, dimBlock >> > (thrust::raw_pointer_cast(d_mesh), thrust::raw_pointer_cast(d_coarse_pairs_cnts), tri_cnt);
		print_cuda_error(__LINE__);
		cudaDeviceSynchronize();

		//int *h_coarse_pairs_cnts = (int*)malloc(sizeof(int)*tri_cnt);//Temp
		//thrust::copy(d_coarse_pairs_cnts, d_coarse_pairs_cnts + tri_cnt, h_coarse_pairs_cnts); //Temp
		//for (int i = 0; i < tri_cnt; i++) {//Temp
		//	printf("%d ", h_coarse_pairs_cnts[i]);//Temp
		//}//Temp
		//printf("\n");//Temp
		int *last_coarse_pairs_cnts_ptr = (int*)malloc(sizeof(int));

		cudaMemcpy(last_coarse_pairs_cnts_ptr, thrust::raw_pointer_cast(d_coarse_pairs_cnts) + tri_cnt - 1, sizeof(int), cudaMemcpyDeviceToHost);

		int last_coarse_pairs_cnts_value = *last_coarse_pairs_cnts_ptr;// d_coarse_pairs_cnts[tri_cnt - 1];
		

		d_indicies = d_coarse_pairs_cnts;
		thrust::exclusive_scan(d_indicies, d_indicies + tri_cnt, d_indicies);
		cudaDeviceSynchronize();

		//int *h_indicies = (int*)malloc(sizeof(int)*tri_cnt);// Temp
		//thrust::copy(d_indicies, d_indicies + tri_cnt, h_indicies); //Temp
		//for (int i = 0; i < tri_cnt; i++) {//Temp
		//	printf("%d ", h_indicies[i]);//Temp
		//}//Temp
		//printf("\n");//Temp
		
		int *last_indicies_ptr = (int*)malloc(sizeof(int));
		cudaMemcpy(last_indicies_ptr, thrust::raw_pointer_cast(d_indicies) + tri_cnt - 1, sizeof(int), cudaMemcpyDeviceToHost);

		int last_indicies_value = *last_indicies_ptr;// d_indicies[tri_cnt - 1];

		total_coarse_pair_cnt = last_indicies_value + last_coarse_pairs_cnts_value;

	}


	// Phase 2 Part 1: allocate coarse_pairs based on total_coarse_pairs_cnt, tag it by 1 based on indicies, transformed into triangles ids using inclusive scan.
	{
		d_coarse_pairs = thrust::device_malloc<int>(total_coarse_pair_cnt);


		thrust::fill(d_coarse_pairs, d_coarse_pairs + total_coarse_pair_cnt, 0);
		cudaDeviceSynchronize();


		dim3 dimBlock(BLOCK_SIZE, 1, 1);
		dim3 dimGrid(ceil(tri_cnt / (float)BLOCK_SIZE), 1, 1);
		printf("\tLaunched tag_segments kernel\n");
		tag_segments << <dimGrid, dimBlock >> >(thrust::raw_pointer_cast(d_coarse_pairs), thrust::raw_pointer_cast(d_indicies), tri_cnt);
		print_cuda_error(__LINE__);
		cudaDeviceSynchronize();


		//int *h_coarse_pairs = (int*) malloc(sizeof(int)*total_coarse_pair_cnt); //Temp
		//thrust::copy(d_coarse_pairs, d_coarse_pairs + total_coarse_pair_cnt, h_coarse_pairs); //Temp
		//for (int i = 0; i < total_coarse_pair_cnt; i++) {//Temp
		//	printf("%d ", h_coarse_pairs[i]);//Temp
		//}//Temp
		//printf("\n");//Temp

		thrust::inclusive_scan( d_coarse_pairs, d_coarse_pairs + total_coarse_pair_cnt, d_coarse_pairs);
		cudaDeviceSynchronize();

		//thrust::copy(d_coarse_pairs, d_coarse_pairs + total_coarse_pair_cnt, h_coarse_pairs); //Temp
		//for (int i = 0; i < total_coarse_pair_cnt; i++) {//Temp
		//	printf("%d ", h_coarse_pairs[i]);//Temp
		//}//Temp
		//printf("\n");//Temp
	}

	// Phase 2 Part 2: evaluate each coarse pair seperatly
	{
		grid_size = h_grid_res.x*h_grid_res.y*h_grid_res.z;

		d_overlap_cnts = thrust::device_malloc<uint>(grid_size);
		thrust::fill(d_overlap_cnts, d_overlap_cnts + grid_size, (unsigned int)(DO_ALLOCATION | REQ_ALLOCATION));
		cudaDeviceSynchronize();

		dim3 dimBlock(BLOCK_SIZE, 1, 1);
		dim3 dimGrid(ceil(total_coarse_pair_cnt / (float)BLOCK_SIZE), 1, 1);
		printf("\tLaunched evaluate_coarse_pairs kernel\n");
		evaluate_coarse_pairs << <dimGrid, dimBlock >> >(thrust::raw_pointer_cast(d_mesh), total_coarse_pair_cnt, thrust::raw_pointer_cast(d_indicies), thrust::raw_pointer_cast(d_coarse_pairs), thrust::raw_pointer_cast(d_overlap_cnts));
		print_cuda_error(__LINE__);
		cudaDeviceSynchronize();

		//int *h_coarse_pairs_2 = (int*)malloc(sizeof(int)*total_coarse_pair_cnt); //temp
		//thrust::copy(d_coarse_pairs, d_coarse_pairs + total_coarse_pair_cnt, h_coarse_pairs_2); //temp
		//for (int i = 0; i < total_coarse_pair_cnt; i++) {//temp
		//	printf("%#010x ", h_coarse_pairs_2[i]);//temp
		//}//temp
		//printf("\n");//temp

		//uint *h_overlap_counts = (uint*) malloc(sizeof(uint)*grid_size); //temp
		//thrust::copy(d_overlap_cnts, d_overlap_cnts + grid_size, h_overlap_counts); //temp
		//for (int i = 0; i < grid_size; i++) {//temp
		//	printf("%#010x ", h_overlap_counts[i]);//temp
		//}//temp
		//printf("\n");//temp	
	}

	// phase 3 : generate triangle list, fill it with triangle ids, replace overlap_cnts with pointers to triangle list
	{
		cudaMemcpyFromSymbol(&total_overlaps, d_total_overlaps_count, sizeof(int));
		d_grid = d_overlap_cnts;
		d_triangle_list = thrust::device_malloc<uint>(total_overlaps);

		dim3 dimBlock(BLOCK_SIZE, 1, 1);
		dim3 dimGrid(ceil(total_coarse_pair_cnt / (float)BLOCK_SIZE), 1, 1);
		printf("\tLaunched extract_grid kernel\n");
		extract_grid << <dimBlock, dimGrid >> >(thrust::raw_pointer_cast(d_mesh), thrust::raw_pointer_cast(d_grid), thrust::raw_pointer_cast(d_triangle_list), thrust::raw_pointer_cast(d_coarse_pairs), thrust::raw_pointer_cast(d_indicies), total_coarse_pair_cnt);
		print_cuda_error(__LINE__);
		cudaDeviceSynchronize();

		cudaFree(thrust::raw_pointer_cast(d_coarse_pairs));
		cudaFree(thrust::raw_pointer_cast(d_indicies));

		//int* h_grid = (int*) malloc(sizeof(int)*grid_size); //Temp
		//thrust::copy(d_grid, d_grid + grid_size, h_grid); //Temp
		//for (uint i = 0; i < grid_size; i++) {//Temp
		//	printf("%d ", h_grid[i]);//Temp
		//}//Temp
		//printf("\n");//Temp

		//int* h_triangle_list = (int*)malloc(sizeof(int)*total_overlaps); //Temp
		//thrust::copy(d_triangle_list, d_triangle_list + total_overlaps, h_triangle_list); //Temp
		//for (int i = 0; i < total_overlaps; i++) {//Temp
		//	printf("%#010x ", h_triangle_list[i]);//Temp
		//}//Temp
		//printf("\n");//Temp

		printf("\n\n");
	}

	*d_grid_result = thrust::raw_pointer_cast(d_grid);
	*d_triangle_list_result = thrust::raw_pointer_cast(d_triangle_list);
	*d_mesh_result = thrust::raw_pointer_cast(d_mesh);
	grid_resolution->x = h_grid_res.x;
	grid_resolution->y = h_grid_res.y;
	grid_resolution->z = h_grid_res.z;
	cwidth->x = h_cwidth.x;
	cwidth->y = h_cwidth.y;
	cwidth->z = h_cwidth.z;

	//TODO check out what allocated memory can be freed
}
