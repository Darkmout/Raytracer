#pragma once
#include "Vec3.h"
#include "device_launch_parameters.h"

class Point
{
public:
	float x, y, z;

	__host__ __device__ Point(void);
	__host__ __device__ Point(float, float, float);
	__host__ __device__ ~Point(void);

	__host__ __device__ Vec3 ToVector(Point);

	//TODO: inline ?
	__host__ __device__ inline Point operator-(Point o){return Point(this->x - o.x, this->y - o.y, this->z - o.z);}
	//__host__ __device__ inline operator Vec3() { return Vec3(this->x, this->y, this->z);}


};

