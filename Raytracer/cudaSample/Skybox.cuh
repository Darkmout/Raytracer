#pragma once
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "Vec3.cuh"

class Skybox
{
public:
	__host__ __device__ Skybox(void)
	{}

	__host__ __device__ ~Skybox(void)
	{}


	__host__ __device__ uchar4 GetColor(Vec3 Ray)
	{
		unsigned char col = (Ray.z + 0.5) * 127;
		return make_uchar4(col, col, col, 0);
	}

};