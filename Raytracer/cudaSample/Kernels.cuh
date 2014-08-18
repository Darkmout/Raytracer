#pragma once
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include "Camera.cuh"
#include "Plane.cuh"
#include "Skybox.cuh"

__global__ void RayKernel(uchar4* const outputImageRGBA, Camera camera, Plane* scene, int sceneCount, Skybox skybox, int numRows, int numCols)
{
	//computing the thread index
	const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;
	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
		return;


	Ray ray = camera.GetRay(thread_2D_pos.x, thread_2D_pos.y, numRows, numCols);
	//printf("thread [%d,%d], rayDirectio %f,%f,%f", thread_2D_pos.x, thread_2D_pos.y, ray.Direction.x, ray.Direction.y,ray.Direction.z);
	//computing the intersection
	float CurrentDistance = FLOAT_MAX;
	Plane* closer = NULL;
	for(int i = 0; i < sceneCount; i++)
	{
		float distance = scene[i].Intersect(ray);
		if(CurrentDistance > distance)
		{
			CurrentDistance = distance;
			closer = &scene[i];
		}
	}

	if(closer)
		outputImageRGBA[thread_1D_pos] = closer->material->Color;
	else
		outputImageRGBA[thread_1D_pos] = skybox.GetColor(ray.Direction);
}


void RayCPU(uchar4* const outputImageRGBA, Camera camera, Plane* scene, int sceneCount, int numRows, int numCols)
{
	//computing the thread index
	uchar4* hostOutput = new uchar4[numRows* numCols]();
	uchar4 colored = make_uchar4(255,0,0,0);
	uchar4 notcolored = make_uchar4(0,0,0,0);


	for(int thread_1D_pos = 0; thread_1D_pos < numRows* numCols; thread_1D_pos++)
	{


		Ray ray = camera.GetRay(thread_1D_pos % numCols, thread_1D_pos / numCols, numRows, numCols);
		//printf("thread [%d,%d], rayDirectio %f,%f,%f", thread_2D_pos.x, thread_2D_pos.y, ray.Direction.x, ray.Direction.y,ray.Direction.z);
		//computing the intersection
		float CurrentDistance = FLOAT_MAX;
		Plane* closer = NULL;
		for(int i = 0; i < sceneCount; i++)
		{
			float distance = scene[i].Intersect(ray);
			if(CurrentDistance > distance)
			{
				CurrentDistance = distance;
				closer = &scene[i];
			}
		}

		if(closer)
			hostOutput[thread_1D_pos] = colored;
		else
			hostOutput[thread_1D_pos] = notcolored;
	}

	checkCudaErrors(cudaMemcpy(outputImageRGBA, hostOutput, sizeof(uchar4) * numRows * numCols, cudaMemcpyHostToDevice));
	delete[] hostOutput;
}