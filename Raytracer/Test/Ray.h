#pragma once

#include "Vec3.h"
#include "device_launch_parameters.h"

class Ray
{
private:

public:
	
	Vec3 Origin, Direction;

	__host__ __device__ Ray(void)
	{
	}

	__host__ __device__ Ray(Vec3 Origin, Vec3 Direction)
	{
		this->Origin = Origin;
		this->Direction = Direction;
	}

	__host__ __device__ ~Ray(void)
	{}
};

