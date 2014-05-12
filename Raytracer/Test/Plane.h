#pragma once
#include "Vec3.h"
#include "Point.h"
#include "Ray.h"
#include "device_launch_parameters.h"

class Plane
{
public:
	Point A, B, C, D;
	Vec3 Normal;

	__host__ __device__ Plane(void);
	__host__ __device__ Plane(Point, Point, Point, Point);
	__host__ __device__~Plane(void);

	__host__ __device__ bool Intersect(Ray);
	__host__ __device__ Point FaceToWorld(int, int);
};

