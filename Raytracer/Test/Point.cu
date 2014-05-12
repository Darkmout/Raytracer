#include "Point.h"


__host__ __device__ Point::Point(void)
{
	this->x = 0;
	this->y = 0;
	this->z = 0;
}

__host__ __device__ Point::Point(float x, float y, float z)
{
	this->x = x;
	this->y = y;
	this->z = z;
}

__host__ __device__ Point::~Point(void)
{
}

__host__ __device__ Vec3 Point::ToVector(Point Direction)
{
	Point result = (Direction - *this );
	return Vec3(result.x, result.y, result.z);
}