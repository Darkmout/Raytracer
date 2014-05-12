#include "Plane.h"


__host__ __device__ Plane::Plane(void)
{
}

__host__ __device__ Plane::Plane(Point A, Point B, Point C, Point D)
{
	this->A = A;
	this->B = B;
	this->C = C;
	this->D = D;

	this->Normal = A.ToVector(B) * A.ToVector(C);
}

__host__ __device__ Plane::~Plane(void)
{
}

__host__ __device__ bool Plane::Intersect(Ray IncidentRay)
{
	return true;
	//	__device__ bool Intersect(Vector vector, Face face)
	//{
	//	//I0 + (I1 - I0)t = P+ (P1 - P0)u + (P2 - P0)v;
	//	face.normal = GetNormalVecteur(face);
	//	float o1 = vector.origin.x;
	//	float o2 = vector.origin.y;
	//	float o3 = vector.origin.z;
	//
	//	float d1 = vector.direction.x;
	//	float d2 = vector.direction.y;
	//	float d3 = vector.direction.z;
	//
	//	float x = face.normal.direction.x;
	//	float y = face.normal.direction.y;
	//	float z = face.normal.direction.z;
	//
	//
	//	float a = face.A.x;
	//	float b = face.A.y;
	//	float c = face.A.z;
	//	float d = - (face.a * face.normal.direction.x) - (face.b * face.normal.direction.y) - (face.c * face.normal.direction.z);
	//
	//	float t = (x * o1 - x * a + y * o2 - y * b + z * o3 - z * c) / (- x * d1 - y * d2 - z * d3);
	//	printf("%f \n", t);
	//	Point result;
	//	result.x = o1 + d1 * t;
	//	result.y = o2 + d2 * t;
	//	result.z = o3 + d3 * t;
	//
	//	return (result.y > face.A.y && result.z > face.A.z);
	//}
}

__host__ __device__ Point Plane::FaceToWorld(int x, int y)
{

	Point result = Point(
		this->A.x*x*y + this->B.x*(1-x)*(y) + this->C.x*(x)*(1-y) + this->D.x*(1-x)*(1-y),
		this->A.y*x*y + this->B.y*(1-x)*(y) + this->C.y*(x)*(1-y) + this->D.y*(1-x)*(1-y),
		this->A.z*x*y + this->B.z*(1-x)*(y) + this->C.z*(x)*(1-y) + this->D.z*(1-x)*(1-y)
	);

	return result;
}