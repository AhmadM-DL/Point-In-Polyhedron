#if 0
#include <stdio.h>
#include <math.h>
#include "pip_helpers.cuh"

void main() {

	// A: 1,0,3
	// B: 2,1,6
	// C: 7,-2,0
	// D: 0,6,0

	float triangle1[9] = { 2, 1, 6, //BAD
						   1, 0, 3,
						   0, 6, 0};

	float triangle2[9] = { 2, 1, 6, //BCA
						   7, -2, 0,
		                   1, 0, 3 };

	float triangle3[9] = { 2, 1, 6, //BDC
						   0, 6, 0, 
		                   7, -2, 0 };

	float triangle4[9] = { 0, 6, 0, //DAC
						   1, 0, 3,
						   7, -2, 0 };

	float ray_origin[3] = {3, 0, 3};

	float iv1 = ray_triangle_intersection(triangle1, ray_origin);
	float iv2 = ray_triangle_intersection(triangle2, ray_origin);
	float iv3 = ray_triangle_intersection(triangle3, ray_origin);
	float iv4 = ray_triangle_intersection(triangle4, ray_origin);


	float summation = iv1+iv2+iv3+iv4;

}
#endif // 0