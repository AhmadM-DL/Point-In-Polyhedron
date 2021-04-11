#include <stdio.h>
#include <math.h>

//Loading an Obj File Required Libs
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

//Cuda Libs
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "helper_math.h"
//#include "GpuTimer.h"

// Random Number Generator
#include <random>

using namespace std;


/**
* Thie flag corresponds to 4 different states of a cell
* empty(11b)(3d) | singualr(10b)(2d) | inside(01b)(1d) | outside(00b)(0d)
*/
#define CELL_STATE_OFFSET 30
#define CELL_STATE ((uint)3<<CELL_STATE_OFFSET)

//#define CELL_STATE_EMPTY ((uint)3<<CELL_STATE_OFFSET)
#define CELL_STATE_SINGULAR ((uint)2<<CELL_STATE_OFFSET)
#define CELL_STATE_INSIDE ((uint)1<<CELL_STATE_OFFSET)
#define CELL_STATE_OUTSIDE ((uint)0<<CELL_STATE_OFFSET)

/**
* The triangle list contains a sublist for each occupied voxel.
* This flag is set in each element that terminates a sublist.
*/
#define TERMINATE_SUBLIST ((uint)1<<31)

#define MAX_SEARCH_UN_SINGULAR 2




// A routine to calculate the dot product of two vertices
// v1 and v2 should be of length 3
inline float __host__ __device__  dot_product(float *v1, float *v2) {
	return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

// A routine to calculate the dot product of two vertices
// v1 and v2 should be of length 3
inline float __host__ __device__  dot_product(double *v1, double *v2) {
	return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

// A routine to calculate the cross product of two vertices
// v1 and v2 should be of length 3
inline void __host__ __device__ cross_product(float *v1, float *v2, float *cross) {
	cross[0] = v1[1] * v2[2] - v1[2] * v2[1];
	cross[1] = -1 * (v1[0] * v2[2] - v1[2] * v2[0]);
	cross[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

// A routin to laod an object (.obj) file into an array of triangle, vertices, numv (number of vertices), and numtri (number of triangles)
void objLoader(char *path, std::vector<int> *triangles_vector, std::vector<float> *vertices_vector, int* numv, int* numtri, float3* min, float3* max) {

	printf("Loading object(.obj) file: %s\n\t", path);
	float minx_v = 0, miny_v = 0, minz_v = 0;
	float maxx_v = 0, maxy_v = 0, maxz_v = 0;

	std::ifstream ifs(path, std::ifstream::in);
	std::string line, key;
	int numv_value = 0, numtri_value = 0;
	int linesReadCounter = 0;
	while (ifs.good() && !ifs.eof() && std::getline(ifs, line)) {
		linesReadCounter++;
		if (linesReadCounter % 2000 == 0) printf(".");

		key = "";
		std::stringstream stringstream(line);
		stringstream >> key >> std::ws;

		if (key == "v") { // vertex
			float x; int counter = 0;
			while (!stringstream.eof()) {
				stringstream >> x >> std::ws;
				switch (counter)
				{
				case 0: (x < minx_v) ? minx_v = x : minx_v;
					(x > maxx_v) ? maxx_v = x : maxx_v;
					break;

				case 1:	(x < miny_v) ? miny_v = x : miny_v;
					(x > maxy_v) ? maxy_v = x : maxy_v;
					break;

				case 2: (x < minz_v) ? minz_v = x : minz_v;
					(x > maxz_v) ? maxz_v = x : maxz_v;
					break;
				default:
					break;
				}
				vertices_vector->push_back(x);
				counter++;
			}
			numv_value++;
		}
		else if (key == "vp") { continue; }
		else if (key == "vt") { continue; }
		else if (key == "vn") { continue; }

		else if (key == "f") { // face
			int v, t, n;
			while (!stringstream.eof()) {
				stringstream >> v >> std::ws;
				triangles_vector->push_back(v - 1);
				if (stringstream.peek() == '/') {
					stringstream.get();
					if (stringstream.peek() == '/') {
						stringstream.get();
						stringstream >> n >> std::ws;
					}
					else {
						stringstream >> t >> std::ws;
						if (stringstream.peek() == '/') {
							stringstream.get();
							stringstream >> n >> std::ws;
						}
					}
				}
			}//Line Stream Loop
			numtri_value++;
		}// IF Line starts with "f"
	}// File Line Stream Loop
	printf("\n\tThe model in the file have : %d trinagles and %d vertices\n", numtri_value, numv_value);
	printf("\tThe model bounding box min. coordinate: (%f,%f,%f)\n", minx_v, miny_v, minz_v);
	printf("\tThe model bounding box max. coordinate: (%f,%f,%f)\n", maxx_v, maxy_v, maxz_v);
	printf("\n\n");

	ifs.close();
	*numv = numv_value; *numtri = numtri_value;
	min->x = minx_v; max->x = maxx_v;
	min->y = miny_v; max->y = maxy_v;
	min->z = minz_v; max->z = maxz_v;
}

// A routine to test whether a ray from an origin point intersects a triangle
// The routine returns a value such that if it is summed will lead to a 0 (no intersection point outside) non-zero (intersection)
float __device__ __host__ ray_triangle_intersection(float3* triangle, float3 ray_origin) {
	// According to  Möller-Trumbore algorithm:
	// Given a triangle ABC and Ray of parametric equation: O + tD (O origin and D direction vector and t parameter)
	// Let E1 = B-A ; E2= C-A ; T= O-A ; P= DxE2 ; Q= TxE1  
	// Then the barycenter coordinates of the intersection point between the ray and the triangle(u,v) 
	// and the parameter t of the ray equation are given by the following equality:
	// |t|      1		 | Q.E2|
	// |u| = _________ * | P.T |
	// |v|     P.E1      | Q.D |

	constexpr float kEpsilon = 1e-8;
	constexpr float PI = 3.14159265;

	// TODO use math helper types float3 int3; for know glue only

	// Triangle Vertices
	//A
	float v0[3] = { triangle[0].x, triangle[0].y, triangle[0].z };
	//B
	float v1[3] = { triangle[1].x, triangle[1].y, triangle[1].z };
	//C
	float v2[3] = { triangle[2].x, triangle[2].y, triangle[2].z };

	// Ray Origin
	float o[3] = { ray_origin.x, ray_origin.y, ray_origin.z };

	//D
	float ray[3] = { 1,0,0 };

	// E1
	float v0v1[3] = { v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2] };
	// E2
	float v0v2[3] = { v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2] };

	// P= DxE2
	float pvec[3];
	cross_product(ray, v0v2, pvec);

	// (DxE2).E1 => P.E1 also equals to D.(E1xE2) => D.N
	float det = dot_product(pvec, v0v1);

	// ray and triangle are parallel if det is close to 0
	if (fabs(det) < kEpsilon) { return 0; }

	// Used to calculate u,v, and t where u,v are barycenter coordinates and t is the distance btween point triangle intersection and ray origin
	float invDet = 1 / det;

	// T = O - A
	float tvec[3] = { o[0] - v0[0], o[1] - v0[1], o[2] - v0[2] };

	// u = (1/det)*P.T
	float u = invDet*dot_product(pvec, tvec);

	// if u <0 or >1 => point doesn't belong to triangle ABC
	if (u < 0 || u > 1) { return 0; }

	// Q = TxE1
	float qvec[3];
	cross_product(tvec, v0v1, qvec);

	// v= (1/det)*Q.D
	float v = invDet*dot_product(qvec, ray);

	// if v <0 or u+v=w(third barycenter coordinate) >1 => point doesn't belong to triangle ABC
	if (v < 0 || u + v > 1) { return 0; }

	// w = 1-(u+v) the last barycentric coordinate
	float w = 1 - (u + v);

	// t = (1/det)*Q.E2
	float t = invDet*dot_product(qvec, v0v2);

	// if t<0 then the triangle is behind the ray => the point doesn't belong to the triangle
	if (t < 0) { return 0; }

	else {// Point inside triangle check if it intersects and edge or a vertex

		  //Is ray entering or leaving based on sign of D.N(Normal of Triangle) => D.(E2xE1) => (DxE2).D => P.D => Det
		  // if Det>0 => entering else leaving
		float in_out_sign = det / fabs(det);

		if (u == 1 || v == 1 || w == 1) {
			// Point intersect a vertex
			// check which vertex
			// Note that a point P is defined using barycenter coordinates as following: wA + uB + vC

			// In case of intersection with vertex V the intersection value will be:  in_out_sign*alpha/2Pi
			// where alpha is the projection of the angle V on the plane perpendicular to ray direction
			// This approach is not totally true as it doesn't cover some corner cases
			// I have to discusses this with Dr. Manal

			if (w == 1) {// Ray intersect A

				float ABProj[3] = { 0, v1[1] - v0[1], v1[2] - v0[2] };
				float ACProj[3] = { 0, v2[1] - v0[1], v2[2] - v0[2] };

				float ABProjLength = sqrt(pow(ABProj[1], 2) + pow(ABProj[2], 2));
				float ACProjLength = sqrt(pow(ACProj[1], 2) + pow(ACProj[2], 2));

				float alpha_radians = acos(dot_product(ABProj, ACProj) / (ABProjLength*ACProjLength));
				float alpha = alpha_radians * 180 / PI;

				return in_out_sign*(alpha / 360);
			}

			if (u == 1) {// Ray intersect B

				float BAProj[3] = { 0, v0[1] - v1[1], v0[2] - v1[2] };
				float BCProj[3] = { 0, v2[1] - v1[1], v2[2] - v1[2] };

				float BAProjLength = sqrt(pow(BAProj[1], 2) + pow(BAProj[2], 2));
				float BCProjLength = sqrt(pow(BCProj[1], 2) + pow(BCProj[2], 2));

				float alpha_radians = acos(dot_product(BAProj, BCProj) / (BAProjLength*BCProjLength));

				float alpha = alpha_radians * 180.0 / PI;

				return in_out_sign*(alpha / 360);

			}
			if (v == 1) {// Ray intersect C

				float CAProj[3] = { 0, v0[1] - v2[1], v0[2] - v2[2] };
				float CBProj[3] = { 0, v1[1] - v2[1], v1[2] - v2[2] };

				float CAProjLength = sqrt(pow(CAProj[1], 2) + pow(CAProj[2], 2));
				float CBProjLength = sqrt(pow(CBProj[1], 2) + pow(CBProj[2], 2));

				float alpha_radians = acos(dot_product(CAProj, CBProj) / (CAProjLength*CBProjLength));
				float alpha = alpha_radians * 180 / PI;

				return in_out_sign*(alpha / 360);
			}

		}

		if (u == 0 || v == 0 || w == 0) {// Point intersect an edge
			return in_out_sign*0.5;
		}

		// Otherwise the ray intersects the body of the triangle 
		return in_out_sign;
	}
}

float __device__ __host__ triangel_area(float* e1, float* e2) {

	float cross[3];
	cross_product(e1, e2, cross);
	return 0.5*sqrt(pow(cross[0], 2) + pow(cross[0], 2) + pow(cross[0], 2));

}

bool __device__ __host__ point_in_triangle(float* v0, float* v1, float* v2, float* point) {

	// Compute vectors        
	float v1v0[3] = { v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2] };
	float v2v0[3] = { v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2] };

	float normal[3];
	cross_product(v1v0, v2v0,normal);

	float normalLength = sqrt(pow(normal[0], 2) + pow(normal[1], 2) + pow(normal[2], 2));

	//Normalize N
	normal[0] /= normalLength;
	normal[1] /= normalLength;
	normal[2] /= normalLength;
	// Project the vector from a triangle's point to a point on the segment to the normal
	float vector[3] = { v0[0] - point[0], v0[1] - point[1], v0[2] - point[2] };

	float projection = dot_product(vector, normal);

	if (projection != 0) {

		//Point and triangle are not coplanar
		return 0;
	}

	//printf("\nPoint and Triangle are coplanar ");


	float proj_point[3] = { point[0] - projection*normal[0], point[1] - projection*normal[1] , point[2] - projection*normal[2] };

	float pv0[3] = { proj_point[0] - v0[0], proj_point[1] - v0[1], proj_point[2] - v0[2] };

	// Compute dot products
	float dot00 = dot_product(v1v0, v1v0);
	float dot01 = dot_product(v1v0, v2v0);
	float dot02 = dot_product(v1v0, pv0);
	float dot11 = dot_product(v2v0, v2v0);
	float dot12 = dot_product(v2v0, pv0);

	// Compute barycentric coordinates
	float invDenom = 1 / (dot00 * dot11 - dot01 * dot01);
	float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
	//printf("\n u=%f",u);
	float v = (dot00 * dot12 - dot01 * dot02) * invDenom;
	//printf("\n v=%f", v);

	// Check if point is in triangle
	return (u >= 0) && (v >= 0) && (u + v <= 1);

}

// A routine to test whether a segment from an origin point and a distance in x_axis intersects a triangle
// The routine returns a value such that if it is summed will lead to a 0 (no intersection point outside) non-zero (intersection)
float __device__ __host__ segment_triangle_intersection(float3* triangle, float3 segment_origin, float segment_length, bool *is_singular) {
	// According to  Möller-Trumbore algorithm:
	// Given a triangle ABC and Ray of parametric equation: O + tD (O origin and D direction vector and t parameter)
	// Let E1 = B-A ; E2= C-A ; T= O-A ; P= DxE2 ; Q= TxE1  
	// Then the barycenter coordinates of the intersection point between the ray and the triangle(u,v) 
	// and the parameter t of the ray equation are given by the following equality:
	// |t|      1		 | Q.E2|
	// |u| = _________ * | P.T |
	// |v|     P.E1      | Q.D |
	//printf("\n\n\n\n");
	constexpr float kEpsilon = 1e-8;
	constexpr float PI = 3.14159265;
	*is_singular = false;

	// TODO use math helper types float3 int3; for now glue only


	// Triangle Vertices
	// A
	float v0[3] = { triangle[0].x, triangle[0].y, triangle[0].z };
	//printf("\nVertex A: %f %f %f", v0[0], v0[1], v0[2]);

	// B
	float v1[3] = { triangle[1].x, triangle[1].y, triangle[1].z };
	//printf("\nVertex B: %f %f %f", v1[0], v1[1], v1[2]);

	// C
	float v2[3] = { triangle[2].x, triangle[2].y, triangle[2].z };
	//printf("\nVertex C: %f %f %f", v2[0], v2[1], v2[2]);

	// Ray Origin
	float o[3] = { segment_origin.x, segment_origin.y, segment_origin.z };

	// segmetn end point we are only moving in x-axis directtion
	float end_point[3] = { segment_origin.x + segment_length, segment_origin.y, segment_origin.z };

	if (point_in_triangle(v0, v1, v2, end_point)) {
		*is_singular=true;
	}

	// D
	float ray[3] = { 1, 0, 0 };

	// E1
	float v0v1[3] = { v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2] };
	// E2
	float v0v2[3] = { v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2] };

	// P= DxE2
	float pvec[3];
	cross_product(ray, v0v2, pvec);
	//printf("\n pvec (%f, %f, %f)", pvec[0], pvec[1], pvec[2]);

	// (DxE2).E1 => P.E1 also equals to D.(E1xE2) => D.N
	float det = dot_product(pvec, v0v1);

	// ray and triangle are parallel if det is close to 0
	if (fabs(det) < kEpsilon) {
		return 0;
	}

	//printf("seg_length=%f ", segment_length);

	// Used to calculate u,v, and t where u,v are barycenter coordinates and t is the distance between point triangle intersection and ray origin
	float invDet = 1 / det;

	// T = O - A
	float tvec[3] = { o[0] - v0[0], o[1] - v0[1], o[2] - v0[2] };

	// u = (1/det)*P.T
	float u = invDet*dot_product(pvec, tvec);
	//printf("\nu=%f ", u);


	// if u <0 or >1 => point doesn't belong to triangle ABC
	if (u < 0 || u > 1) {
		//printf("\nSegment Intersection o %f %f %f: point doesn't belong to triangle ABC u=%f", o[0], o[1], o[2], u);
		return 0;
	}

	// Q = TxE1
	float qvec[3];
	cross_product(tvec, v0v1, qvec);

	// v= (1/det)*Q.D
	float v = invDet*dot_product(qvec, ray);
	//printf(" v=%f ", v);


	// if v <0 or u+v=w(third barycenter coordinate) >1 => point doesn't belong to triangle ABC
	if (v < 0 || u + v > 1) {
		//printf("\nSegment Intersection o %f %f %f: point doesn't belong to triangle ABC v=%f w=%f", o[0], o[1], o[2], v, 1 - u - v);
		return 0;
	}

	// w = 1-(u+v) the last barycentric coordinate
	float w = 1 - (u + v);
	//printf(" w=%f ", w);


	// t = (1/det)*Q.E2
	float t = invDet*dot_product(qvec, v0v2);
	//printf(" t=%f ", t);

	//if t<0 then the triangle is behind the ray => the point doesn't belong to the triangle
	if (t < -kEpsilon) {
		//printf("\nSegment Intersection o %f %f %f: triangle is behind the ray t=%f", o[0], o[1], o[2], t);
		return 0;
	}

	// if t>segment then the triangle is after the segment end point 
	if (t > segment_length) {
		//printf("\nSegment Intersection o %f %f %f: triangle is after the segment end point t=%f segment=%f", o[0], o[1], o[2], t, segment_length);
		return 0;
	}

	//else Point inside triangle check if it intersects and edge or a vertex

		  //Is ray entering or leaving based on sign of D.N(Normal of Triangle) => D.(E2xE1) => (DxE2).D => P.D => Det
		  // if Det>0 leaving else entering
	float in_out_sign = det / fabs(det);

	if (u == 1 || v == 1 || w == 1) {
		// Point intersect a vertex
		// check which vertex
		// Note that a point P is defined using barycenter coordinates as following: wA + uB + vC

		// In case of intersection with vertex V the intersection value will be:  in_out_sign*alpha/2Pi
		// where alpha is the projection of the angle V on the plane perpendicular to ray direction
		// This approach is not totally true as it doesn't cover some corner cases
		// I have to discusses this with Dr. Manal

		if (w == 1) {// Ray intersect A

			float ABProj[3] = { 0, v1[1] - v0[1], v1[2] - v0[2] };
			float ACProj[3] = { 0, v2[1] - v0[1], v2[2] - v0[2] };

			float ABProjLength = sqrt(pow(ABProj[1], 2) + pow(ABProj[2], 2));
			float ACProjLength = sqrt(pow(ACProj[1], 2) + pow(ACProj[2], 2));

			float alpha_radians = acos(dot_product(ABProj, ACProj) / (ABProjLength*ACProjLength));
			float alpha = alpha_radians * 180 / PI;

			//printf("\nSegment Intersection o %f %f %f: Ray intersect A w=%f", o[0], o[1], o[2], w);
			return in_out_sign*(alpha / 360);
		}

		if (u == 1) {// Ray intersect B

			float BAProj[3] = { 0, v0[1] - v1[1], v0[2] - v1[2] };
			float BCProj[3] = { 0, v2[1] - v1[1], v2[2] - v1[2] };

			float BAProjLength = sqrt(pow(BAProj[1], 2) + pow(BAProj[2], 2));
			float BCProjLength = sqrt(pow(BCProj[1], 2) + pow(BCProj[2], 2));

			float alpha_radians = acos(dot_product(BAProj, BCProj) / (BAProjLength*BCProjLength));

			float alpha = alpha_radians * 180.0 / PI;

			//printf("\nSegment Intersection o %f %f %f: Ray intersect B u=%f", o[0], o[1], o[2], u);
			return in_out_sign*(alpha / 360);

		}
		if (v == 1) {// Ray intersect C

			float CAProj[3] = { 0, v0[1] - v2[1], v0[2] - v2[2] };
			float CBProj[3] = { 0, v1[1] - v2[1], v1[2] - v2[2] };

			float CAProjLength = sqrt(pow(CAProj[1], 2) + pow(CAProj[2], 2));
			float CBProjLength = sqrt(pow(CBProj[1], 2) + pow(CBProj[2], 2));

			float alpha_radians = acos(dot_product(CAProj, CBProj) / (CAProjLength*CBProjLength));
			float alpha = alpha_radians * 180 / PI;

			//printf("\nSegment Intersection o %f %f %f: Ray intersect C v=%f", o[0], o[1], o[2], v);
			return in_out_sign*(alpha / 360);
		}

	}

	if (u == 0 || v == 0 || w == 0) {// Point intersect an edge
		//printf("\nSegment Intersection o %f %f %f: Point intersect an edge", o[0], o[1], o[2]);
		return in_out_sign*0.5;
	}

	// Otherwise the ray intersects the body of the triangle 
	//printf("\nSegment Intersection o %f %f %f: ray intersects the body of the triangle ", o[0], o[1], o[2]);
	return in_out_sign;
	//}
}


// A routine to test whether a segment from an origin point and a distance in x_axis intersects a triangle
// The routine returns a value such that if it is summed will lead to a 0 (no intersection point outside) non-zero (intersection)
float __device__ __host__ un_algined_segment_triangle_intersection(float3* triangle, float3 segment_origin, float3 segment_end) {

	// According to  Möller-Trumbore algorithm:
	// Given a triangle ABC and Ray of parametric equation: O + tD (O origin and D direction vector and t parameter)
	// Let E1 = B-A ; E2= C-A ; T= O-A ; P= DxE2 ; Q= TxE1  
	// Then the barycenter coordinates of the intersection point between the ray and the triangle(u,v) 
	// and the parameter t of the ray equation are given by the following equality:
	// |t|      1		 | Q.E2|
	// |u| = _________ * | P.T |
	// |v|     P.E1      | Q.D |
	//printf("\n\n\n\n");
	constexpr float kEpsilon = 1e-8;
	constexpr float PI = 3.14159265;

	// TODO use math helper types float3 int3; for now glue only

	//printf("segment_origin=(%f, %f, %f) ", segment_origin.x, segment_origin.y, segment_origin.z);
	//printf("segment_end=(%f, %f, %f) ", segment_end.x, segment_end.y, segment_end.z);

	// Triangle Vertices
	//A
	float v0[3] = { triangle[0].x, triangle[0].y, triangle[0].z };
	//printf("Vertex A: %f %f %f ", v0[0], v0[1], v0[2]);

	//B
	float v1[3] = { triangle[1].x, triangle[1].y, triangle[1].z };
	//printf("Vertex B: %f %f %f ", v1[0], v1[1], v1[2]);

	//C
	float v2[3] = { triangle[2].x, triangle[2].y, triangle[2].z };
	//printf("Vertex c: %f %f %f ", v2[0], v2[1], v2[2]);

	// Ray Origin
	float o[3] = { segment_origin.x, segment_origin.y, segment_origin.z };

	//D 
	float ray[3] = { segment_end.x - segment_origin.x,  segment_end.y - segment_origin.y,  segment_end.z - segment_origin.z };
	//printf("ray=(%f, %f, %f) ", ray[0], ray[1], ray[2]);

	//Segment Size
	float segment_length = sqrt(pow(ray[0], 2) + pow(ray[1], 2) + pow(ray[2], 2));

	// Normalize Ray
	ray[0] /= segment_length; ray[1] /= segment_length; ray[2] /= segment_length;

	// E1
	float v0v1[3] = { v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2] };
	//printf("v0v1=(%f, %f, %f) ", v0v1[0], v0v1[1], v0v1[2]);

	// E2
	float v0v2[3] = { v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2] };
	//printf("v0v2=(%f, %f, %f) ", v0v2[0], v0v2[1], v0v2[2]);

	// E3
	float v1v2[3] = { v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2] };

	// P= DxE2
	float pvec[3];
	cross_product(ray, v0v2, pvec);
	//printf("pvec=(%f, %f, %f) ", pvec[0], pvec[1], pvec[2]);

	// (DxE2).E1 => P.E1 also equals to D.(E1xE2) => D.N
	float det = dot_product(pvec, v0v1);
	//printf("det=%f ", det);

	// ray and triangle are parallel if det is close to 0
	if (fabs(det) < kEpsilon) {
		//printf("\nSegment Intersection o %f %f %f: ray and triangle are parallel det=%f", o[0], o[1], o[2], det);
		return 0;
	}
	//printf("seg_length=%f ", segment_length);


	// Used to calculate u,v, and t where u,v are barycenter coordinates and t is the distance btween point triangle intersection and ray origin
	float invDet = 1 / det;
	//printf("invDet=%f ", det);


	// T = O - A
	float tvec[3] = { o[0] - v0[0], o[1] - v0[1], o[2] - v0[2] };
	//printf(" tvec=(%f, %f, %f)", tvec[0], tvec[1], tvec[2]);


	// u = (1/det)*P.T
	float u = invDet*dot_product(pvec, tvec);
	//printf("u=%f ", u);


	// if u <0 or >1 => point doesn't belong to triangle ABC
	if (u < 0 || u > 1) {
		//printf("\nSegment Intersection o %f %f %f: point doesn't belong to triangle ABC u=%f", o[0], o[1], o[2], u);
		return 0;
	}

	// Q = TxE1
	float qvec[3];
	cross_product(tvec, v0v1, qvec);
	//printf(" qvec=(%f, %f, %f)", qvec[0], qvec[1], qvec[2]);


	//v= (1/det)*Q.D
	float v = invDet*dot_product(qvec, ray);
	//printf("v=%f ", v);


	// if v <0 or u+v=w(third barycenter coordinate) >1 => point doesn't belong to triangle ABC
	if (v < 0 || u + v > 1) {
		//printf("\nSegment Intersection o %f %f %f: point doesn't belong to triangle ABC v=%f w=%f", o[0], o[1], o[2], v, 1-u-v);
		return 0;
	}

	// w = 1-(u+v) the last barycentric coordinate
	float w = 1 - (u + v);
	//printf("w=%f ", w);


	// t = (1/det)*Q.E2
	float t = invDet*dot_product(qvec, v0v2);
	//printf("t=%f ", t);

	//if t<0 then the triangle is behind the ray => the point doesn't belong to the triangle
	if (t < 0) {
		//printf("\nSegment Intersection o %f %f %f: triangle is behind the ray t=%f", t);
		return 0;
	}

	// if t>segment then the triangle is after the segment end point 
	if (t > segment_length) {
		//printf("\nSegment Intersection o %f %f %f: triangle is after the segment end point t=%f segment=%f", o[0], o[1], o[2], t, segment_length);
		return 0;
	}

	//else Point inside triangle check if it intersects and edge or a vertex

	//Is ray entering or leaving based on sign of D.N(Normal of Triangle) => D.(E2xE1) => (DxE2).D => P.D => Det
	// if Det>0 => entering else leaving
	float in_out_sign = det / fabs(det);

	if (u==1 || v==1 || w==1) {
		// Point intersect a vertex
		// check which vertex
		// Note that a point P is defined using barycenter coordinates as following: wA + uB + vC

		// Based on the paper in case of intersection with vertex V the intersection value will be:  in_out_sign*alpha/2Pi
		// where alpha is the projection of the angle V on the plane perpendicular to ray direction
		// This approach is not totally true as it doesn't cover some corner cases
		// I have to discusses this with Dr. Manal

		if (w==1) {// Ray intersect A

			double ratioAB = dot_product(v0v1, ray) / pow(segment_length, 2);
			double ABProj[3] = { v0v1[0] - ratioAB*ray[0], v0v1[1] - ratioAB*ray[1], v0v1[2] - ratioAB*ray[2] };

			double ratioAC = dot_product(v0v2, ray) / pow(segment_length, 2);
			double ACProj[3] = { v0v2[0] - ratioAC*ray[0], v0v2[1] - ratioAC*ray[1], v0v2[2] - ratioAC*ray[2] };

			double ABProjLength = sqrt(pow(ABProj[0], 2) + pow(ABProj[1], 2) + pow(ABProj[2], 2));
			double ACProjLength = sqrt(pow(ACProj[0], 2) + pow(ACProj[1], 2) + pow(ACProj[2], 2));

			double alpha_radians = acos(dot_product(ABProj, ACProj) / (ABProjLength*ACProjLength));
			double alpha = alpha_radians * 180 / PI;

			//printf("\nSegment Intersection o %f %f %f: Ray intersect A w=%f", o[0], o[1], o[2], w);
			return in_out_sign*(alpha / 360);
		}

		if (u == 1) {// Ray intersect B

			double ratioBA = -dot_product(v0v1, ray) / pow(segment_length, 2);
			double BAProj[3] = { -v0v1[0] - ratioBA*ray[0], -v0v1[1] - ratioBA*ray[1], -v0v1[2] - ratioBA*ray[2] };

			double ratioBC = dot_product(v1v2, ray) / pow(segment_length, 2);
			double BCProj[3] = { v1v2[0] - ratioBC*ray[0], v1v2[1] - ratioBC*ray[1], v1v2[2] - ratioBC*ray[2] };

			double BAProjLength = sqrt(pow(BAProj[0], 2) + pow(BAProj[1], 2) + pow(BAProj[2], 2));
			double BCProjLength = sqrt(pow(BCProj[0], 2) + pow(BCProj[1], 2) + pow(BCProj[2], 2));

			double alpha_radians = acos(dot_product(BAProj, BCProj) / (BAProjLength*BCProjLength));

			double alpha = alpha_radians * 180.0 / PI;

			//printf("\nSegment Intersection o %f %f %f: Ray intersect B u=%f", o[0], o[1], o[2], u);
			return in_out_sign*(alpha / 360);

		}
		if (v==1) {// Ray intersect C

			double ratioCA = -dot_product(v0v2, ray) / pow(segment_length, 2);
			double CAProj[3] = { -v0v2[0] - ratioCA*ray[0], -v0v2[1] - ratioCA*ray[1], -v0v2[2] - ratioCA*ray[2] };

			double ratioCB = -dot_product(v1v2, ray) / pow(segment_length, 2);
			double CBProj[3] = { -v1v2[0] - ratioCB*ray[0], -v1v2[1] - ratioCB*ray[1], -v1v2[2] - ratioCB*ray[2] };

			double CAProjLength = sqrt(pow(CAProj[0], 2) + pow(CAProj[1], 2) + pow(CAProj[2], 2));
			double CBProjLength = sqrt(pow(CBProj[0], 2) + pow(CBProj[1], 2) + pow(CBProj[2], 2));

			double alpha_radians = acos(dot_product(CAProj, CBProj) / (CAProjLength*CBProjLength));
			double alpha = alpha_radians * 180 / PI;

			//printf("\nSegment Intersection o %f %f %f: Ray intersect C v=%f", o[0], o[1], o[2], v);
			return in_out_sign*(alpha / 360);
		}

	}

	if (u==0|| v==0 || w==0) {// Point intersect an edge
		//printf("\nSegment Intersection o %f %f %f: Point intersect an edge",o[0], o[1], o[2]);
		return in_out_sign*0.5;
	}

	// Otherwise the ray intersects the body of the triangle 
	//printf("\nSegment Intersection o %f %f %f: ray intersects the body of the triangle ", o[0], o[1], o[2]);
	return in_out_sign;
	//}
}

// A routine to transform vertices/triangles arrays into one mesh array
void toMesh(int* triangles, float* vertices, int numtri, std::vector<float3> *mesh_vector) {

	for (int i = 0; i < numtri; i++) {

		//Add coordinates of first vertex
		int v1_index = triangles[3 * i];
		mesh_vector->push_back(make_float3(vertices[3 * v1_index], vertices[3 * v1_index + 1], vertices[3 * v1_index + 2]));

		//Add coordinates of second vertex
		int v2_index = triangles[3 * i + 1];
		mesh_vector->push_back(make_float3(vertices[3 * v2_index], vertices[3 * v2_index + 1], vertices[3 * v2_index + 2]));

		//Add coordinates of third vertex
		int v3_index = triangles[3 * i + 2];
		mesh_vector->push_back(make_float3(vertices[3 * v3_index], vertices[3 * v3_index + 1], vertices[3 * v3_index + 2]));
	}
}

// A routine to print a triangle from the triangles list given it's index
void printTriangle(int *triangles, float *vertices, int triangleIndex) {

	int ptrv1 = triangles[3 * triangleIndex];
	int ptrv2 = triangles[3 * triangleIndex + 1];
	int ptrv3 = triangles[3 * triangleIndex + 2];

	float v1[3] = { vertices[3 * ptrv1], vertices[3 * ptrv1 + 1], vertices[3 * ptrv1 + 2] };
	float v2[3] = { vertices[3 * ptrv2], vertices[3 * ptrv2 + 1], vertices[3 * ptrv2 + 2] };
	float v3[3] = { vertices[3 * ptrv3], vertices[3 * ptrv3 + 1], vertices[3 * ptrv3 + 2] };

	printf("Triangle (Face) %d points to vertices %d, %d, and %d\n", triangleIndex, ptrv1, ptrv2, ptrv3);
	printf("It's vertices are:\n");
	printf("V(%d): (%f, %f, %f)\n", ptrv1, v1[0], v1[1], v1[2]);
	printf("V(%d): (%f, %f, %f)\n", ptrv2, v2[0], v2[1], v2[2]);
	printf("V(%d): (%f, %f, %f)\n\n", ptrv3, v3[0], v3[1], v3[2]);
	return;
}

// A routine to print a triangle from a mesh given it's index
void printTriangle(float3* mesh, int triangle) {
	int triIndex = triangle * 3 * 3;

	float3 v0 = mesh[3 * triIndex];
	float3 v1 = mesh[3 * triIndex + 1];
	float3 v2 = mesh[3 * triIndex + 2];

	printf("Triangle (Face) %d is defined by the following vertices:\n", triangle);
	printf("Vertex 0: (%f, %f, %f)\n", v0.x, v0.y, v0.y);
	printf("Vertex 1: (%f, %f, %f)\n", v1.x, v1.y, v1.y);
	printf("Vertex 2: (%f, %f, %f)\n\n", v2.x, v2.y, v2.y);
}

uint __device__ __host__ compute_globally_cell_state(uint *grid, uint cell_idx, float cell_intersection_value, bool is_singular) {

	if (! (cell_intersection_value == 1 || cell_intersection_value == 0 || cell_intersection_value == -1)) {
		printf("compute_globally_cell_state: Intersection value is not regular %f", cell_intersection_value);
	}

	bool even_intersection = ((int)cell_intersection_value % 2 == 0);

	//printf("\nCell %u have and intersection value equall to %f", cell_idx,cell_intersection_value);

	if (is_singular) {
		grid[cell_idx] |= CELL_STATE_SINGULAR;
		//printf("\nCell %u Is singular = %u\n", cell_idx, grid[cell_idx]);
		return CELL_STATE_SINGULAR;
	}

	if (even_intersection) {
		grid[cell_idx] |= CELL_STATE_OUTSIDE;
		//printf("\nCell %u Is outside = %u\n", cell_idx, grid[cell_idx]);
		return CELL_STATE_OUTSIDE;
	}

	if (!even_intersection) {
		grid[cell_idx] |= CELL_STATE_INSIDE;
		//printf("\nCell %u Is inside = %u\n", cell_idx, grid[cell_idx]);
		return CELL_STATE_INSIDE;
	}

}

uint __device__ compute_locally_point_state(uint origin_point_inclusion_state, float cell_intersection_value) {

	if (!(cell_intersection_value == 1 || cell_intersection_value == 0 || cell_intersection_value == -1)) {
		printf("compute_locally_point_state: Intersection value is not regular %f", cell_intersection_value);
	}

	bool even_intersection = ((int)cell_intersection_value % 2 == 0);

	if (origin_point_inclusion_state == CELL_STATE_INSIDE && even_intersection)return CELL_STATE_INSIDE;

	if (origin_point_inclusion_state == CELL_STATE_INSIDE && !even_intersection)return CELL_STATE_OUTSIDE;

	if (origin_point_inclusion_state == CELL_STATE_OUTSIDE && even_intersection)return CELL_STATE_OUTSIDE;

	if (origin_point_inclusion_state == CELL_STATE_OUTSIDE && !even_intersection)return CELL_STATE_INSIDE;
	
}

bool __device__ triangle_in_array(uint triangle, uint* triangles, uint triangles_length) {

	for (int i = 0; i < triangles_length; i++) {
		if (triangles[i] == triangle) return true;
	}
	return false;
}

__host__ inline void print_if_cuda_error(int line) {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("\nError On Line %d: %s\n", line - 1, cudaGetErrorString(err));
	}
}


void __device__ get_triangles_overlap_2_cell(uint cell_id, int cell_2_id, uint* grid, uint* grid_end, uint* triangle_list, uint** triangles, uint* triangles_size) {

	// Triangle list 1 start and end
	uint t_l_1_start;
	int t_l_1_end;

	// Triangle list 2 start and end
	uint t_l_2_start;
	int t_l_2_end;

	// cell 1 is empty
	// set start to 0 and end to -1
	// this will lead to a size of 0
	// the triangles loop won't be entered
	if (grid_end[cell_id] & ~CELL_STATE - grid[cell_id] & ~CELL_STATE == 0) {
		t_l_1_start = 0;
		t_l_1_end = -1;
	}
	else {
		t_l_1_start = grid[cell_id] & ~CELL_STATE;
		t_l_1_end = grid_end[cell_id] & ~CELL_STATE;
	}

	// cell 2 is empty or not provided
	if ((cell_2_id == -1) || (grid_end[cell_2_id] & ~CELL_STATE - grid[cell_2_id] & ~CELL_STATE == 0)) {
		t_l_2_start = 0;
		t_l_2_end = -1;
	}
	else {
		t_l_2_start = grid[cell_2_id] & ~CELL_STATE;
		t_l_2_end = grid_end[cell_2_id] & ~CELL_STATE;
	}

	// Compute lists sizes
	uint t_l_1_size = t_l_1_end - t_l_1_start + 1;
	uint t_l_2_size = t_l_2_end - t_l_2_start + 1;

	// if both cells are empty
	if (t_l_1_size == 0 && t_l_2_size == 0) {
		*triangles_size = 0;
		return;
	}

	// Triangle List
	*triangles = new uint[t_l_1_size + t_l_2_size];

	if ((*triangles) == NULL) {
		printf("\nError: no enough dynamic memory in the heap to be allocated\n");
	}

	for (uint i = 0; i < t_l_1_size; i++) {
		(*triangles)[i] = triangle_list[t_l_1_start + i] & ~TERMINATE_SUBLIST;
	}

	int new_tri = 0;
	for (uint i = 0; i < t_l_2_size; i++) {

		if (!triangle_in_array(triangle_list[t_l_2_start + i] & ~TERMINATE_SUBLIST, *triangles, t_l_1_size)) {
			(*triangles)[t_l_1_size + new_tri] = triangle_list[t_l_2_start + i] & ~TERMINATE_SUBLIST;
			new_tri++;
		}

	}

	*triangles_size = t_l_1_size + new_tri;

	return;
}

char __device__ get_triangles_overlap_cells(uint* cell_start, uint* cell_end, uint cell_ids_size, uint* triangle_list, uint** triangles, uint* triangles_size) {

	// Triangle list starts, ends
	uint *t_l_start = new uint[cell_ids_size];
	uint *t_l_end = new uint[cell_ids_size];

	uint max_triangles_size = 0;
	uint actuall_num_triangles = 0;

	for (uint i = 0; i < cell_ids_size; i++) {
		t_l_start[i] = cell_start[i] & ~CELL_STATE;
		t_l_end[i] = cell_end[i] & ~CELL_STATE;
	}

	uint size=0;
	for (uint i = 0; i < cell_ids_size; i++) {
		size = (cell_end[i] & ~CELL_STATE) - (cell_start[i] & ~CELL_STATE);
		if (size == 0) max_triangles_size += 0;
		else max_triangles_size += size + 1; // + 1 as pointers starts from 0
	}

	if (max_triangles_size == 0) { // No triangles overlap the specified cells
		*triangles_size = 0;
		delete t_l_start;
		delete t_l_end;
		return 1;
	}

	// allocate a temporery triangles array based on max_triangles_size
	uint *triangles_temp = (uint*)malloc(sizeof(uint)*max_triangles_size);

	if (!triangles_temp) {
		printf("\nError(get_triangles_overlap_cells): no enough dynamic memory in the heap to be allocated\n");
		return 0;
	}

	// loop on the cells 
	for (uint i = 0; i < cell_ids_size; i++) {

		if (t_l_end[i] - t_l_start[i] == 0) { // Cell is empty; continue to nex cells
			continue;
		}

		// loop on triangles overlapping the cell
		for (uint j = 0; j < t_l_end[i] - t_l_start[i] + 1; j++) {

			// if triangle not already in the triangles list, add it
			if (!triangle_in_array(triangle_list[t_l_start[i] + j] & ~TERMINATE_SUBLIST, triangles_temp, actuall_num_triangles)) {
				triangles_temp[actuall_num_triangles] = triangle_list[t_l_start[i] + j] & ~TERMINATE_SUBLIST;
				actuall_num_triangles++;
			}
		}
	}

	// Now allocate the actual number of triangels to the triangles list
	*triangles = (uint* )malloc(sizeof(uint)*actuall_num_triangles);

	if (!(*triangles)) {
		printf("\nError(get_triangles_overlap_cells): no enough dynamic memory in the heap to be allocated\n");
		return 0;
	}

	// loop on triangles_lsit and add them to the triangles array
	for (uint i = 0; i < actuall_num_triangles; i++) {
		(*triangles)[i] = triangles_temp[i];
	}

	free(triangles_temp);
	delete t_l_start;
	delete t_l_end;
	*triangles_size = actuall_num_triangles;

	return 1;
}


void generate_random_points(float minx, float maxx, float miny, float maxy, float minz, float maxz, uint numpoints, std::vector<float3> *points_vector) {
	std::random_device rd;
	std::mt19937 e2(rd());
	std::uniform_real_distribution<> distx(minx, maxx);
	std::uniform_real_distribution<> disty(miny, maxy);
	std::uniform_real_distribution<> distz(minz, maxz);

	for (int n = 0; n < numpoints; ++n) {
		points_vector->push_back(make_float3(distx(e2), disty(e2), distz(e2)));
	}
}

void generate_points(float minx, float maxx, float miny, float maxy, float minz, float maxz, uint numpoints, std::vector<float3> *points_vector) {

	float xstep = (maxx - minx) / numpoints;
	float ystep = (maxy - miny) / numpoints;
	float zstep = (maxz - minz) / numpoints;

	float accx = 0;
	float accy = 0;
	float accz = 0;
	for (int n = 0; n < numpoints; ++n) {
		points_vector->push_back(make_float3(accx+xstep, accy + ystep, accz + zstep));
		accx +=xstep;
		accy += ystep;
		accz += zstep;
	}
}


void export_grid_points(uint* grid, uint gridSize, float3 min, float3 max, uint3 grid_res, float3 cell_width) {
	ofstream inside_point_stream;
	ofstream outside_point_stream;
	ofstream singular_point_stream;

	inside_point_stream.open("inside_grid_points.obj");
	outside_point_stream.open("outside_grid_points.obj");
	singular_point_stream.open("singular_grid_points.obj");

	uint cell_id;
	uint cell_state;
	uint3 cell;
	float3 center_point;

	 
	for (uint i = 0; i < gridSize; i++) {
		
		cell_state = grid[i] & CELL_STATE;
		cell_id = i;

		cell = { cell_id % grid_res.x, (cell_id / grid_res.x) % grid_res.y, ((cell_id / grid_res.x) / grid_res.y)};
		//cell = { ((cell_id / grid_res.x) / grid_res.y) ,  (cell_id / grid_res.x) % grid_res.y , cell_id % grid_res.x };
		//cell = { cell_id % grid_res.x, ((cell_id / grid_res.x) / grid_res.y), (cell_id / grid_res.x) % grid_res.y };

		center_point={ min.x + cell.x*cell_width.x + cell_width.x / 2, min.y + cell.y*cell_width.y + cell_width.y / 2, min.z + cell.z*cell_width.z + cell_width.z / 2 };

		if (cell_state == CELL_STATE_INSIDE) {
			inside_point_stream << center_point.x << " " << center_point.y << " " << center_point.z << " " << "\n";
			continue;
		}

		if (cell_state == CELL_STATE_OUTSIDE) {
			outside_point_stream << center_point.x << " " << center_point.y << " " << center_point.z << " " << "\n";
			continue;
		}

		if (cell_state == CELL_STATE_SINGULAR) {
			singular_point_stream << center_point.x << " " << center_point.y << " " << center_point.z << " " << "\n";
			continue;
		}
		
	}
	inside_point_stream.close();
	outside_point_stream.close();
	singular_point_stream.close();


}

void export_test_points_as_obj_files(float3 *points, char *points_inclusion, uint numpoints) {

		ofstream inside_point_stream;
		ofstream outside_point_stream;


		uint inside_points_count = 0;
		uint outside_points_count = 0;

		inside_point_stream.open("inside_points.obj");
		outside_point_stream.open("outside_points.obj");

		for (uint i = 0; i < numpoints; i++){
			if (points_inclusion[i] == (char) 1) {
				inside_point_stream <<points[i].x<<" "<<points[i].y<<" "<<points[i].z<<" "<<"\n";
				inside_points_count++;
				continue;
			}

			if (points_inclusion[i] == (char) 0) {
				outside_point_stream << points[i].x << " " << points[i].y << " " << points[i].z << " " << "\n";
				outside_points_count++;
				continue;
			}
			//else
			printf("Point (%f,%f,%f) have a strange inclusion state %d", points[i].x, points[i].y, points[i].z, points_inclusion[i]);
		}


		inside_point_stream.close();
		outside_point_stream.close();

		printf("Exporting Results\n");
		printf("\tExported %u inside-points to file inside_points.obj and %u outside-points to file outside_points.obj\n", inside_points_count, outside_points_count);
		printf("\n\n");
}