#pragma once

#include "cuda.h"
#include "cuda_gl_interop.h"
#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>

void CheckCudaError(cudaError_t cudaStatus)
{
	if(cudaStatus != cudaSuccess)
	{
		printf(cudaGetErrorString(cudaStatus));
		exit(1);
	}
}
