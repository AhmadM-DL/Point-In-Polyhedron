![](docs/record.gif)

# Point-In-Polyhedron

Point in polyhedron is an important problem that have a number of applications in different domains such as computer aided design (CAD), computer graphics, geographic information systems (GIS), etc. [1]

With the rise of GPGPU (General Purpose Graphics Processing Unit) many geometric algorithms have been redesigned and implemented to work on GPUs in order to achieve a better performance. In this repo. I combine and reimplement 2 research papers trying to parallelize the procedure on GPU using CUDA.

A [https://ahmadm-dl.github.io/Point-In-Polyhedron/](three.js demo is available here)

---

## Polyhedron

"In geometry, a polyhedron (plural polyhedra or polyhedrons) is a three-dimensional shape with flat polygonal faces, straight edges and sharp corners or vertices. A polyhedron is a 3-dimensional example of the more general polytope in any number of dimensions." Wikipedia.

"In computational geometry, polygon triangulation is the decomposition of a polygonal area (simple polygon) P into a set of triangles, finding a set of triangles with pairwise non-intersecting interiors whose union is P." Wikipedia.

In computer graphics 3d models are represeanted by a mech of triangles forming a polyhedron.

## Ray crossing

Ray crossing a well-known inclusion test. The idea is to create a ray from the query point and count the number of intersections with the polyhedron faces (or polygon edges). If the intersection number is even then the point is out of the polyhedron if it is odd then the point is inside (see figure below). The test soundness is based on Jordan Curve Theorem.

## Parallel Algorithm

In this section the algorithm proposed in [1] is explained. The explanation will be limited to the general procedures without mentioning tiny details as interested readers are advised to check the paper in [1]. Also illustrations of different steps will make use of polygons rather than polyhedrons for convenience without losing generality.

The proposed algorithm for point in polyhedron test can be divided into three steps.

 1. Grid generation and initialization step
 2. Preprocessing step
 3. Point in polyhedron test step

For further details check [docs\Project Report.pdf](report).

### References

[1]	L. Jing and W. Wencheng, "Fast and robust GPU-based point-in-polyhedron determination," Computer Aided Design, 2017.

[2]	R. R. J. Antonio and O. M. Lidia, "Geometric Algorithms on CUDA," in GRAPP, Funchal, Madeira - Portugal , 2008.

[3]	F. Feito and J. Torres, "Inclusion test for general polyhedra," Computers & Graphics, vol. 21, no. 1, pp. 23-30, 1997.

[4]	T. I. M. Eugene and P. N. Sumanta, "Macro 64-regions for uniform grids on GPU," Vis Comput, vol. 30, pp. 615-624, 2014.
