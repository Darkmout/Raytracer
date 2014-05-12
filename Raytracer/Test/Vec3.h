#pragma once
//#include "Point.h"
#include "device_launch_parameters.h"

class Vec3
{
public:
	float x, y, z;
	__host__ __device__ Vec3(void);
	__host__ __device__ Vec3(float, float, float);
	__host__ __device__ ~Vec3(void);

	__host__ __device__ inline Vec3 operator* (Vec3 o){ return Vec3(	this->y * o.z - this->z * o.y, this->z * o.x - this->x * o.z, this->x * o.y - this->y * o.x	);}
	//__host__ __device__ inline operator Point() { return Point(this->x, this->y, this->z);}

};

