#pragma once
#include "device_launch_parameters.h"

#include "Point.h"
#include "Plane.h"
#include "Ray.h"

class Camera
{
public:
	int Width, Height;
	Point Origin;
	Plane Plan;

	__host__ __device__ Camera(void);
	__host__ __device__ Camera(int Width,int Height);
	__host__ __device__ ~Camera(void);

	__host__ __device__ Ray GetRay(int, int);
};

