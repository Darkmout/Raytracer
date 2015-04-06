#pragma once
//#include "device_launch_parameters.h"
#include <cuda_runtime.h>
//#define GL_GLEXT_PROTOTYPES
//#include "GL/glut.h"

#include <GL/glew.h>
#include <GL/freeglut.h>
#include "cuda.h"
#include "cuda_gl_interop.h"

#include "CudaUtils.cuh"
#include "Model.cuh"
#include "Camera.cuh"
#include "Plane.cuh"
#include "Vec3.cuh"
#include "mtllib.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string>



//  Avoid showing up the console window
//#pragma comment(linker,"/subsystem:\"windows\" /entry:\"mainCRTStartup\"")

//  constants representing the window size
#define WINDOW_WIDTH 512
#define WINDOW_HEIGHT 512

//the scene
Model scene;

//  Represents the pixel buffer in memory
GLuint bufferObj; //OpenGL's buffer of the data
cudaGraphicsResource *resource; //CUDA's buffer for the data
GLubyte buffer[WINDOW_WIDTH][WINDOW_HEIGHT][3];
uchar4 *d_outputImageRGBA, *h_outputImageRGBA;


int oldMouseX;
int oldMouseY;
bool mouseClick = false;






//kernel
__global__ void RayKernel(uchar4* const outputImageRGBA,Camera camera, Plane* scene, int sceneCount, int numRows, int numCols)
{
	//computing the thread index
	const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;
	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
		return;


	Ray ray = camera.GetRay(thread_2D_pos.x, thread_2D_pos.y, numRows, numCols);
	//printf("thread [%d,%d], rayDirectio %f,%f,%f", thread_2D_pos.x, thread_2D_pos.y, ray.Direction.x, ray.Direction.y,ray.Direction.z);
	//computing the intersection
	bool intersect = false;
	for(int i = 0; i < sceneCount; i++)
	{
		if(scene[i].Intersect(ray))
			intersect = true;
	}

	if(intersect)
		outputImageRGBA[thread_1D_pos] = make_uchar4(255,255,255, 255);
	else
		outputImageRGBA[thread_1D_pos] = make_uchar4(0,0,0, 255);

}

const dim3 blockSize(16 , 16);
const dim3 gridSize (WINDOW_WIDTH / blockSize.x + 1, WINDOW_HEIGHT / blockSize.y + 1);

void generateImage ()
{
	cudaEvent_t startTimer, stopTimer;
	cudaEventCreate(&startTimer);
	cudaEventCreate(&stopTimer);
	cudaEventRecord(startTimer,0);

	RayKernel<<<gridSize, blockSize>>>(d_outputImageRGBA, scene.camera, scene.d_scene, scene.Planes.size(), WINDOW_HEIGHT, WINDOW_WIDTH);	cudaDeviceSynchronize(); CheckCudaError(cudaGetLastError());

	cudaEventRecord(stopTimer,0);
	cudaEventSynchronize(startTimer);
	cudaEventSynchronize(stopTimer);
	float timerResult;
	cudaEventElapsedTime(&timerResult, startTimer, stopTimer);

	glutSetWindowTitle(std::to_string(100./timerResult).c_str());

}






//-------------------------------------------------------------------------
//  This function is passed to the glutMouseFunc and is called 
//  whenever the mouse is clicked.
//-------------------------------------------------------------------------
void mouseMove (int x, int y)
{
	printf("%d %d\n", oldMouseX-x, oldMouseY-y);
	if(mouseClick)
	{
		scene.camera.Rotation(-(oldMouseX -x), (oldMouseY - y));
		oldMouseX = x;
		oldMouseY = y;
		glutPostRedisplay();

	}
}
void mouseButton(int button, int state, int x, int y)
{
	if(state == GLUT_DOWN)
	{
		mouseClick = true;
		oldMouseX = x;
		oldMouseY = y;
	}
	else
	{

		mouseClick = false;
	}
}

//-------------------------------------------------------------------------
//  This function is passed to the glutKeyboardFunc and is called s
//  whenever the user hits a key.
//-------------------------------------------------------------------------
void keyboard (unsigned char key, int x, int y)
{
	//switch (key)
	//{
	//case 'z':
	//	scene.camera.Move(Vec3(1,0,0));
	//	break;
	//case 'q':
	//	scene.camera.Move(Vec3(0,-1,0));
	//	break;
	//case 's':
	//	scene.camera.Move(Vec3(-1,0,0));
	//	break;
	//case 'd':
	//	scene.camera.Move(Vec3(0,1,0));
	//	break;
	//case 'c':
	//	scene.camera.Move(Vec3(0,0,-1));
	//	break;
	//case 32:
	//	scene.camera.Move(Vec3(0,0,1));
	//	break;
	//case 27:
	//	exit (0);
	//}
	//glutPostRedisplay();
}




void init ()
{	
	h_outputImageRGBA = (uchar4*)malloc(sizeof(uchar4) * WINDOW_WIDTH * WINDOW_HEIGHT);
	//CheckCudaError(cudaMalloc(&d_outputImageRGBA,   sizeof(uchar4) * WINDOW_WIDTH * WINDOW_HEIGHT));

	glClearColor (0.0, 0.0, 0.0, 0.0);
	glShadeModel(GL_FLAT);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	srand (time(NULL));
	generateImage();
}

//-------------------------------------------------------------------------
//  This function is passed to glutDisplayFunc in order to display 
//	OpenGL contents on the window.
//-------------------------------------------------------------------------
void display (void)
{
	generateImage();
	//CheckCudaError(cudaMemcpy(h_outputImageRGBA, d_outputImageRGBA, sizeof(uchar4) * WINDOW_WIDTH * WINDOW_HEIGHT, cudaMemcpyDeviceToHost));
	int i, j;
	for (i = 0; i < WINDOW_WIDTH; i++) 
	{
		for (j = 0; j < WINDOW_HEIGHT; j++)
		{
			buffer[i][j][0] = (GLubyte) (h_outputImageRGBA[i + j* WINDOW_HEIGHT].x);
			buffer[i][j][1] = (GLubyte) (h_outputImageRGBA[i + j* WINDOW_HEIGHT].y);
			buffer[i][j][2] = (GLubyte) (h_outputImageRGBA[i + j* WINDOW_HEIGHT].z);
		}
	}

	glClear(GL_COLOR_BUFFER_BIT);
	glDrawPixels(WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGB,
		GL_UNSIGNED_BYTE, buffer);
	glutSwapBuffers ();


}


int main (int argc, char* argv[])
{

	//Initialize the OpenGL Driver with GLUT
	glutInitWindowSize (WINDOW_WIDTH, WINDOW_HEIGHT);
	glutInitDisplayMode (GLUT_RGB | GLUT_DOUBLE);
	glutInit(&argc, argv);
	glutCreateWindow ("Raytracer =)");
	//glutFullScreen ();

	//selet device and set it to be used by OpenGL
	cudaDeviceProp prop;
	int device;
	memset(&prop, 0 , sizeof(cudaDeviceProp));
	prop.major = 2;
	prop.minor = 0;
	CheckCudaError(cudaChooseDevice(&device, &prop));
	CheckCudaError(cudaGLSetGLDevice(device));

	//Load the scene
	scene = Model(std::string(argv[argc-1]));

	
	//set the interoperation between CUDA and GLUT with a buffer used by booth APIs
	glGenBuffers(1, &bufferObj);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, WINDOW_WIDTH * WINDOW_HEIGHT * 4, NULL, GL_DYNAMIC_DRAW_ARB);
	//  Set OpenGL program initial state.
	init();


	// Set the callback functions
	glutDisplayFunc (display);
	glutKeyboardFunc (keyboard);
	glutMotionFunc(mouseMove);
	glutMouseFunc(mouseButton);

	glutWarpPointer(WINDOW_WIDTH/2, WINDOW_HEIGHT/2);

	

	//  Start GLUT event processing loop
	glutMainLoop();

	return 0;
}