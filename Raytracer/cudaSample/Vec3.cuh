#pragma once
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "Matrix3.cuh"

class Vec3
{
public:
	float x, y, z;

	__host__ __device__ Vec3(void)
	{
		this->x = 0;
		this->y = 0;
		this->z = 0;
	}
	__host__ __device__ Vec3(float x, float y, float z)
	{
		this->x = x;
		this->y = y;
		this->z = z;
	}

	__host__ __device__ ~Vec3(void)
	{}


	__host__ __device__ float Length()
	{
		return sqrt(x*x + y*y + z*z);
	}


	__host__ __device__ void Normalize()
	{
		float sum = this->Length();
		this->x /= sum;
		this->y /= sum;
		this->z /= sum; 
	}


	__host__ __device__  Vec3 operator-(Vec3 o){
		return Vec3(this->x - o.x, this->y - o.y, this->z - o.z);
	}

	__host__ __device__  Vec3 operator+(Vec3 o){
		return Vec3(this->x + o.x, this->y + o.y, this->z + o.z);
	}

	__host__ __device__  Vec3 operator*(float o){
		return Vec3(this->x * o, this->y * o, this->z * o);
	}

	//TODO: tis is equal to [3x1] * [3x3] 
	__host__ __device__  Vec3 operator*(Matrix3 o){
		return Vec3(
			this->x * o.aa + this->y * o.ab + this->z * o.ac, 
			this->x * o.ba + this->y * o.bb + this->z * o.bc, 
			this->x * o.ca + this->y * o.cb + this->z * o.cc
			);
	}

	__host__ __device__  Vec3 operator/(float o){
		return Vec3(this->x / o, this->y / o, this->z / o);
	}

	//TODO: plus lisible?
	__host__ __device__ Vec3 Cross (Vec3 o)
	{ 
		return Vec3(
			this->y * o.z - this->z * o.y,
			this->z * o.x - this->x * o.z, 
			this->x * o.y - this->y * o.x);
	}




	//dot product of two perpendicular vectors = 0
	//dot product of two vectors pointing to the same direction > 0
	//dot product of two vectors pointing to opposite directions < 0
	__host__ __device__ float Dot (Vec3 o)
	{ 
		return 	this->x * o.x + this->y * o.y + this->z * o.z ;
	}

};

