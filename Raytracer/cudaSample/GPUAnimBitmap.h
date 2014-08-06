

// OpenGL Graphics includes
#include <GL/glew.h>
#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
// Sorry for Apple : unsigned int sampler is not available to you, yet...
// Let's switch to the use of PBO and glTexSubImage
#define USE_TEXSUBIMAGE2D
#else
#include <GL/freeglut.h>
#endif

#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "device_launch_parameters.h"

#include <iostream>
#include <cstdio>
#include <ctime>


class GPUAnimBitmap
{
public:

	//  Represents the pixel buffer in memory
	GLuint BufferObj; //OpenGL's buffer of the data
	cudaGraphicsResource *Resource; //CUDA's buffer for the data
	uchar4* devPtr;
	size_t size;
	int Width, Height;
	void (*DrawCallback)(uchar4*, int, int);
	
	GPUAnimBitmap(int argc, char** argv,  void (*DrawCallback)(uchar4*, int, int) = NULL, void (*KeyboardCallback)(unsigned char, int, int) = NULL, void (*MouseCallback)(int, int, int, int) = NULL);
	void GPUAnimBitmap::StartMainLoop();
	~GPUAnimBitmap();
};

// static method used for glut callbacks
static GPUAnimBitmap** get_bitmap_ptr( void ) {
	static GPUAnimBitmap*   gBitmap;
	return &gBitmap;
}


static void GLDraw(void)
{
	int time = glutGet(GLUT_ELAPSED_TIME);

	GPUAnimBitmap*  bitmap = *(get_bitmap_ptr());
	uchar4*         devPtr;
	size_t  size;

	checkCudaErrors( cudaGraphicsMapResources( 1, &(bitmap->Resource), NULL ) );
	checkCudaErrors( cudaGraphicsResourceGetMappedPointer( (void**)&devPtr, &size, bitmap->Resource) );

	bitmap->DrawCallback(devPtr, bitmap->Width, bitmap->Height);
	
	checkCudaErrors( cudaGraphicsUnmapResources( 1, &(bitmap->Resource), NULL ) );

	glDrawPixels( bitmap->Width, bitmap->Height, GL_RGBA, GL_UNSIGNED_BYTE, 0 );
	glutSwapBuffers();

	int FPS = 1000 / (glutGet(GLUT_ELAPSED_TIME) - time);
	glutSetWindowTitle(std::to_string(FPS).c_str());
}


GPUAnimBitmap::GPUAnimBitmap(int argc, char** argv, void (*DrawCallback)(uchar4*, int, int),  void (*KeyboardCallback)(unsigned char, int, int), void (*MouseCallback)(int, int, int, int))
{
	GPUAnimBitmap**   bitmap = get_bitmap_ptr();
	*bitmap = this;

	Width = 512;
	Height = 512;

	this->DrawCallback = DrawCallback;

	//selet device and set it to be used by OpenGL
	cudaDeviceProp prop;
	int device;
	memset(&prop, 0 , sizeof(cudaDeviceProp));
	prop.major = 2;
	prop.minor = 0;
	checkCudaErrors(cudaChooseDevice(&device, &prop));
	checkCudaErrors(cudaGLSetGLDevice(device));

	//Initialize the OpenGL Driver with GLUT
	glutInit(&argc, argv);
	glutInitWindowSize (Width, Height);
	glutInitDisplayMode (GLUT_DOUBLE | GLUT_RGBA);
	glutCreateWindow ("Raytracer =)");
	//glutFullScreen ();

	// initialize necessary OpenGL extensions
	glewInit();

	if (! glewIsSupported(
		"GL_VERSION_2_0 "
		"GL_ARB_pixel_buffer_object "
		"GL_EXT_framebuffer_object "
		))
	{
		printf("ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		std::exit(-1);
	}


	//set the interoperation between CUDA and GLUT with a buffer used by booth APIs
	glGenBuffers(1, &BufferObj);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, BufferObj);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, Width * Height * 4, NULL, GL_DYNAMIC_DRAW_ARB);

	//notify cuda to share the buffer with both cuda and OpenGL
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&Resource, BufferObj, cudaGraphicsMapFlagsNone));

	//map the shared resource and request a pointer to the mapped resource
	checkCudaErrors( cudaGraphicsMapResources( 1, &Resource, NULL ) );
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer( (void**)&devPtr,&size,Resource ));

	checkCudaErrors(cudaGraphicsUnmapResources(1, &Resource, NULL));
	glutDisplayFunc(GLDraw);
	glutKeyboardFunc(KeyboardCallback);
	glutMouseFunc(MouseCallback);
}

void GPUAnimBitmap::StartMainLoop()
{
		glutMainLoop();
}

GPUAnimBitmap::~GPUAnimBitmap()
{
	checkCudaErrors( cudaGraphicsUnregisterResource( Resource ) );
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );
	glDeleteBuffers( 1, &BufferObj );
}


