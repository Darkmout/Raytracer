#pragma once
#include "Point.h"
#include "Vec3.h"
#include "device_launch_parameters.h"

class Ray
{
public:
	Point Origin;
	Vec3 Direction;

	__host__ __device__ Ray(void);
	__host__ __device__ Ray(Point, Point);
	__host__ __device__ Ray(Point, Vec3);
	__host__ __device__ ~Ray(void);
};

