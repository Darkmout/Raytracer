#include "Vec3.h"


__host__ __device__ Vec3::Vec3(void)
{
	this->x = 0;
	this->y = 0;
	this->z = 0;
}

__host__ __device__ Vec3::Vec3(float x, float y, float z)
{
	this->x = x;
	this->y = y;
	this->z = z;
}


__host__ __device__ Vec3::~Vec3(void)
{
}


