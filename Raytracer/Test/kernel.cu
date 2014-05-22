#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include "cuda_gl_interop.h"
//#include "cuda.h"

#include "Camera.h"
#include "Plane.h"
#include "Vec3.h"
#include "Model.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string>
#include <gl/glut.h>

//  Avoid showing up the console window
//#pragma comment(linker,"/subsystem:\"windows\" /entry:\"mainCRTStartup\"")

//  constants representing the window size
#define WINDOW_WIDTH 512
#define WINDOW_HEIGHT 512

void CheckCudaError(cudaError_t cudaStatus)
{
	if(cudaStatus != cudaSuccess)
	{
		printf(cudaGetErrorString(cudaStatus));
		exit(1);
	}
}

//the scene
Model scene;

//  Initialization
void init ();

//  Support Functions
void centerOnScreen ();

//  define the window position on screen
int window_x;
int window_y;
int oldMouseX;
int oldMouseY;
bool mouseClick = false;

//  variable representing the window title
char *window_title = "Image Generator";

//  Tells whether to display the window full screen or not
//  Press Alt + Esc to exit a full screen.
int full_screen = 0;

//  Generates a random image...
void generateImage ();

//  Represents the pixel buffer in memory
GLubyte buffer[WINDOW_WIDTH][WINDOW_HEIGHT][3];
uchar4 *d_outputImageRGBA, *h_outputImageRGBA;



//-------------------------------------------------------------------------
//  Set OpenGL program initial state.
//-------------------------------------------------------------------------
void init ()
{	
	h_outputImageRGBA = (uchar4*)malloc(sizeof(uchar4) * WINDOW_WIDTH * WINDOW_HEIGHT);
	CheckCudaError(cudaMalloc(&d_outputImageRGBA,   sizeof(uchar4) * WINDOW_WIDTH * WINDOW_HEIGHT));

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

	glClear(GL_COLOR_BUFFER_BIT);
	glDrawPixels(WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGB,
		GL_UNSIGNED_BYTE, buffer);
	glutSwapBuffers ();
	

}



//-------------------------------------------------------------------------
//  This function is passed to the glutMouseFunc and is called 
//  whenever the mouse is clicked.
//-------------------------------------------------------------------------
void mouseMove (int x, int y)
{
	//printf("%d %d\n", oldMouseX-x, oldMouseY-y);
	if(mouseClick)
	{
		scene.camera.Rotation(oldMouseX -x, oldMouseY - y);
		generateImage();
		glutPostRedisplay();
		oldMouseX = x;
		oldMouseY = y;
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
	switch (key)
	{
	case 'z':
		scene.camera.Move(Vec3(1,0,0));
		break;
	case 'q':
		scene.camera.Move(Vec3(0,-1,0));
		break;
	case 's':
		scene.camera.Move(Vec3(-1,0,0));
		break;
	case 'd':
		scene.camera.Move(Vec3(0,1,0));
		break;
	case 'c':
		scene.camera.Move(Vec3(0,0,-1));
		break;
	case 32:
		scene.camera.Move(Vec3(0,0,1));
		break;
	case 27:
		exit (0);
	}
	generateImage();
	glutPostRedisplay();
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

	//outputImageRGBA[thread_1D_pos] = make_uchar4(((ray.Direction.x + 1) / 2) * 255,((ray.Direction.x + 1) / 2)  * 255, ((ray.Direction.x + 1) / 2)  * 255, 255);
}

//-------------------------------------------------------------------------
//  Generate new image with random colors
//-------------------------------------------------------------------------


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
//  Program Main method.
//-------------------------------------------------------------------------
int main (int argc, char* argv[])
{
	scene = Model(std::string(argv[argc-1]));

	//TODO : use Cuda Interop
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
	glutPassiveMotionFunc(mouseMove);
	glutMouseFunc(mouseButton);

	glutWarpPointer(WINDOW_WIDTH/2, WINDOW_HEIGHT/2);

	//  Start GLUT event processing loop
	glutMainLoop();

	return 0;
}