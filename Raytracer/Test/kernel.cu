#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Camera.h"
#include "Point.h"
#include "Plane.h"
#include "Vec3.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <gl/glut.h>

//  Avoid showing up the console window
#pragma comment(linker,"/subsystem:\"windows\" /entry:\"mainCRTStartup\"")

//  constants representing the window size
#define WINDOW_WIDTH 512
#define WINDOW_HEIGHT 512

//  Initialization
void init ();

//  Callback functions
void display (void);
void mouse (int button, int state, int x, int y);
void keyboard (unsigned char key, int x, int y);

//  Support Functions
void centerOnScreen ();

//  define the window position on screen
int window_x;
int window_y;

//  variable representing the window title
char *window_title = "Image Generator";

//  Tells whether to display the window full screen or not
//  Press Alt + Esc to exit a full screen.
int full_screen = 0;

//  Generates a random image...
void generateImage ();

//  Represents the pixel buffer in memory
GLubyte buffer[WINDOW_WIDTH][WINDOW_HEIGHT][3];




//-------------------------------------------------------------------------
//  Set OpenGL program initial state.
//-------------------------------------------------------------------------
void init ()
{	
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
	glClear(GL_COLOR_BUFFER_BIT);
	glDrawPixels(WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGB,
		GL_UNSIGNED_BYTE, buffer);
	glutSwapBuffers ();
}



//-------------------------------------------------------------------------
//  This function is passed to the glutMouseFunc and is called 
//  whenever the mouse is clicked.
//-------------------------------------------------------------------------
void mouse (int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		generateImage ();
		glutPostRedisplay ();
	}
}

//-------------------------------------------------------------------------
//  This function is passed to the glutKeyboardFunc and is called 
//  whenever the user hits a key.
//-------------------------------------------------------------------------
void keyboard (unsigned char key, int x, int y)
{
	switch (key)
	{
	case 'g':
		generateImage ();
		glutPostRedisplay ();
		break;
	case 27:
		exit (0);
	}
}

//-------------------------------------------------------------------------
//  This function sets the window x and y coordinates
//  such that the window becomes centered
//-------------------------------------------------------------------------
void centerOnScreen ()
{
	window_x = (glutGet (GLUT_SCREEN_WIDTH) - WINDOW_WIDTH)/2;
	window_y = (glutGet (GLUT_SCREEN_HEIGHT) - WINDOW_HEIGHT)/2;
}

void CheckCudaError(cudaError_t cudaStatus)
{
	if(cudaStatus != cudaSuccess)
	{
		printf(cudaGetErrorString(cudaStatus));
		exit(1);
	}
}



//kernel
__global__ void RayKernel(uchar4* const outputImageRGBA,Camera camera , Plane plane, int numRows, int numCols)
{
	//computing the thread index
	const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;
	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
		return;


	Ray ray = camera.GetRay(thread_2D_pos.x, thread_2D_pos.y);

	//computing the intersection
	if(plane.Intersect(ray))
		outputImageRGBA[thread_1D_pos] = make_uchar4(255,255,255, 255);
	else
		outputImageRGBA[thread_1D_pos] = make_uchar4(100,0,0, 255);

}

//-------------------------------------------------------------------------
//  Generate new image with random colors
//-------------------------------------------------------------------------
void generateImage ()
{


	const dim3 blockSize(16 , 16);
	const dim3 gridSize (WINDOW_WIDTH / blockSize.x + 1, WINDOW_HEIGHT / blockSize.y + 1);

	//image
	uchar4 *d_outputImageRGBA, *h_outputImageRGBA;
	h_outputImageRGBA = (uchar4*)malloc(sizeof(uchar4) * WINDOW_WIDTH * WINDOW_HEIGHT);
	CheckCudaError(cudaMalloc(&d_outputImageRGBA,   sizeof(uchar4) * WINDOW_WIDTH * WINDOW_HEIGHT));

	//elements

	Camera camera = Camera(WINDOW_WIDTH, WINDOW_HEIGHT);

	Plane plane = Plane(
		Vec3(3,0.5,0.5),
		Vec3(3,0.5,-0.5),
		Vec3(3,-0.5,-0.5),
		Vec3(3,-0.5,0.5)
		);

	RayKernel<<<gridSize, blockSize>>>(d_outputImageRGBA, camera, plane, WINDOW_HEIGHT, WINDOW_WIDTH);	cudaDeviceSynchronize(); CheckCudaError(cudaGetLastError());

	CheckCudaError(cudaMemcpy(h_outputImageRGBA, d_outputImageRGBA, sizeof(uchar4) * WINDOW_WIDTH * WINDOW_HEIGHT, cudaMemcpyDeviceToHost));

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
}


//-------------------------------------------------------------------------
//  Program Main method.
//-------------------------------------------------------------------------
int main (int argc, char **argv)
{
	//  Connect to the windowing system
	glutInit(&argc, argv);

	//  create a window with the specified dimensions
	glutInitWindowSize (WINDOW_WIDTH, WINDOW_HEIGHT);

	//  Set the window x and y coordinates such that the 
	//  window becomes centered
	centerOnScreen ();

	//  Position Window
	glutInitWindowPosition (window_x, window_y);

	//  Set Display mode
	glutInitDisplayMode (GLUT_RGB | GLUT_DOUBLE);

	//  Create window with the specified title
	glutCreateWindow (window_title);

	//  View in full screen if the full_screen flag is on
	if (full_screen)
		glutFullScreen ();

	//  Set OpenGL program initial state.
	init();

	// Set the callback functions
	glutDisplayFunc (display);
	glutKeyboardFunc (keyboard);
	glutMouseFunc (mouse);

	//  Start GLUT event processing loop
	glutMainLoop();

	return 0;
}