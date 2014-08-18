// CUDA includes
#include "GPUAnimBitmap.h"
#include "Kernels.cuh"
#include "Model.cuh"

#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

#include <rendercheck_gl.h>

#include <stdio.h>
#include <stdlib.h>


Model Scene;


void DrawCallback(uchar4* devPtr, int Width, int Height)
{
	//RayCPU(devPtr, Scene.camera, &Scene.Planes[0], Scene.Planes.size(), Height, Width);
	dim3 blockSize(16 , 16);
	dim3 gridSize (Width / blockSize.x + 1, Height / blockSize.y + 1);

	RayKernel<<<gridSize, blockSize>>>(devPtr, Scene.camera, Scene.d_scene, Scene.Planes.size(), Skybox(), Height, Width);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

void KeyboardCallback(unsigned char key, int x, int y)
{
	switch (key)
	{
	case '1':
		Scene.camera.ToFront();
		break;
	case '2':
		Scene.camera.ToOrigin();
		break;
	case '3':
		Scene.camera.ToUp();
		break;
	case 'z':
		Scene.camera.Move(Vec3(1,0,0));
		break;
	case 'q':
		Scene.camera.Move(Vec3(0,-1,0));
		break;
	case 's':
		Scene.camera.Move(Vec3(-1,0,0));
		break;
	case 'd':
		Scene.camera.Move(Vec3(0,1,0));
		break;
	case 'c':
		Scene.camera.Move(Vec3(0,0,-1));
		break;
	case 32:
		Scene.camera.Move(Vec3(0,0,1));
		break;
	case 27:
		exit (0);
	}
	glutPostRedisplay();
}

int oldX, oldY;

void MouseCallback(int button, int state, int x, int y)
{
	if(state == GLUT_DOWN)
	{
		oldX = x;
		oldY = y;
	}
}
void MotionCallback(int x, int y)
{
		Scene.camera.Rotation(oldX-x, oldY-y);
		oldX = x;
		oldY = y;
		glutPostRedisplay();
}

int main (int argc, char** argv)
{

	//Create a Window (has to be before all cuda call)
	GPUAnimBitmap Drawer = GPUAnimBitmap(argc, argv, &DrawCallback, &KeyboardCallback, &MouseCallback, &MotionCallback);

	//Load the scene
	Scene = Model(std::string(argv[argc-1]));

	//start GLutMaintLoop
	Drawer.StartMainLoop();

	return 0;
}