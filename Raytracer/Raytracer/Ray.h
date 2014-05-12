#pragma once
#include "Point.h"
#include "Vector.h"
#include "device_launch_parameters.h"

class Ray
{
public:
	Point Origin;
	Vector Direction;

	__host__ __device__ Ray(void);
	__host__ __device__ Ray(Point, Point);
	__host__ __device__ Ray(Point, Vector);
	__host__ __device__ ~Ray(void);
};

