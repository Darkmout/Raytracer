#pragma once
#include "device_launch_parameters.h"
#include "cuda_runtime.h"

class Material
{
public:
	//const char* Name;
	uchar4 Color;

	__host__ __device__ Material(void)
	{}

	__host__ __device__ ~Material(void)
	{}


	__host__ __device__ Material(uchar4 Color)
	{
		this->Color = Color;
	}

};