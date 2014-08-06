#pragma once
#include "device_launch_parameters.h"
#include "cuda_runtime.h"

class mtl
{
public:
	const char* name;

	__host__ __device__ mtl(void)
	{}

	__host__ __device__ ~mtl(void)
	{}



	__host__ __device__ mtl(const char* name)
	{
		this->name = name;
	}

};