#pragma once
#include "Point.h"
#include "device_launch_parameters.h"

class Vector
{
public:
	float x, y, z;
	__host__ __device__ Vector(void);
	__host__ __device__ Vector(float, float, float);
	__host__ __device__ ~Vector(void);

	__host__ __device__ inline Vector operator* (Vector o){ return Vector(	this->y * o.z - this->z * o.y, this->z * o.x - this->x * o.z, this->x * o.y - this->y * o.x	);}
	//__host__ __device__ inline operator Point() { return Point(this->x, this->y, this->z);}

};

