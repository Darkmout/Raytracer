#pragma once
#include "Vector.h"
#include "device_launch_parameters.h"

class Point
{
public:
	float x, y, z;

	__host__ __device__ Point(void);
	__host__ __device__ Point(float, float, float);
	__host__ __device__ ~Point(void);

	//TODO: inline ?
	__host__ __device__ inline Point operator-(Point o){return Point(this->x - o.x, this->y - o.y, this->z - o.z);}
	//__host__ __device__ inline operator Vector() { return Vector(this->x, this->y, this->z);}

	__host__ __device__ Vector To(Point);
};

