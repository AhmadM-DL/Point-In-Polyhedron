#if 0
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

//Thrust
#include <thrust\scan.h>
#include <thrust\device_ptr.h>
#include <thrust\device_malloc.h>
#include <thrust\fill.h>

#define BLOCK_SIZE 1024
#define NUM_POINTS 100000
#define MAX_BLOCK_NUMBERS 65535
//TODO add texture support

// Well launch blocks equal to number of points and threads 1024
// If number of points is larger than the max possible number of blocks 
// use max 
__global__ void PointInPolyhedron0(float3* mesh, float3 *points, char* points_out, uint numpoints, uint numtri) {
	
	__shared__ float point_intersection_value;

	// In case the number of points is more than the number of blocks each block will 
	// work on multiple point
	
	int point_id =  blockIdx.x ;
	
	for (int point_id = blockIdx.x; point_id < numpoints; point_id+=gridDim.x){

		// Initilize point_intersection_value
		if (threadIdx.x == 0) {
			point_intersection_value = 0;
		}

		// Let other threads wait unitl initlization is done
		__syncthreads();

		// Get Point
		float3 point = points[point_id];

		float thread_intersections_sum = 0;

		//Each thread will compute the intersection of p with numtri/blockdim.x triangles 
		for (int i = threadIdx.x; i < numtri; i += blockDim.x) {
			thread_intersections_sum += ray_triangle_intersection(&mesh[3 * i], point);
		}

		// Write to the corresponding point 
		atomicAdd(&point_intersection_value, thread_intersections_sum);

		// Wait until all threads have done their part of work
		__syncthreads();

		// Transform intersection value into (inside(1)/outside(0)) and write it to global memory
		if (threadIdx.x == 0) {
			//if intersections are even i.e = 0 then the point is outside
			if ((int)point_intersection_value % 2 == 0) {
				points_out[point_id] = 0;
			}
			else { // otherwise ht e point is inside
				points_out[point_id] = 1;
			}
		}

		// Don't go to next loop until the point state is written
		__syncthreads();
	}
}


/**
* Host main routine
*/
int main(void){


	float3 *h_mesh = NULL;
	float3 *h_points = NULL;
	char  *h_points_out= NULL;
	GpuTimer timer;

	thrust::device_ptr<float3> d_mesh; 
	thrust::device_ptr<float3> d_points;
	thrust::device_ptr<char> d_points_out;

	int numv, numtri;
	float3 min;
	float3 max;

	//Load obj file model as a mesh
	//TODO tranform into function
	std::vector<float3> mesh_vector;
	std::vector<float> vertices_vector;
	std::vector<int> triangles_vector;
	char *filepath = "media/dragon.obj";

	objLoader(filepath, &triangles_vector, &vertices_vector, &numv, &numtri, &min, &max);

	toMesh(triangles_vector.data(), vertices_vector.data(), numtri, &mesh_vector);

	h_mesh = mesh_vector.data();

	// Generate Random Points in the polyhedron bounding box
	std::vector<float3> h_points_vector;
	uint numpoints = NUM_POINTS;
	h_points_out = (char*)malloc(sizeof(char)*numpoints);

	generate_random_points(min.x, max.x, min.y, max.y, min.z, max.z, numpoints, &h_points_vector);
	h_points = h_points_vector.data();;

	//Allocate memory for the  device
	d_mesh = thrust::device_malloc<float3>(3*numtri);
	d_points = thrust::device_malloc<float3>(numpoints);
	d_points_out = thrust::device_malloc<char>(numpoints);
	
	//Copyfrom host to device
	thrust::copy(h_mesh, h_mesh + 3*numtri, d_mesh);
	thrust::copy(h_points, h_points + numpoints, d_points);

	//Define Grid Configuration
	dim3 dimBlock(BLOCK_SIZE, 1, 1);
	dim3 dimGrid(fmin(numpoints, MAX_BLOCK_NUMBERS), 1, 1);



	//Launch PointInPolyhedron0
	timer.Start();
	PointInPolyhedron0<<<dimGrid, dimBlock>>>(thrust::raw_pointer_cast(d_mesh), thrust::raw_pointer_cast(d_points),thrust::raw_pointer_cast(d_points_out), numpoints, numtri);
	cudaDeviceSynchronize();
    print_if_cuda_error(__LINE__);
	timer.Stop();
	printf("\t\n Kernel Time: %f msecs.\n", timer.Elapsed());

	// copy from device to host
	thrust::copy(d_points_out, d_points_out + numpoints, h_points_out);
	//export_test_points_as_obj_files(h_points, h_points_out, numpoints);

	cudaFree(thrust::raw_pointer_cast(d_points));
	cudaFree(thrust::raw_pointer_cast(d_points_out));
	cudaFree(thrust::raw_pointer_cast(d_mesh));

	free(h_points);
	free(h_points_out);
	free(h_mesh);

}
#endif // 0

