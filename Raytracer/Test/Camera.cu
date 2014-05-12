#include "Camera.h"


__host__ __device__ Camera::Camera(void)
{
	this->Origin = Point(0,0,0);
	this->Plan = Plane(
		Point(1,0.5,0.5),
		Point(1,0.5,-0.5),
		Point(1,-0.5,-0.5),
		Point(1,-0.5,0.5)
		);
}

__host__ __device__ Camera::Camera(int Width, int Height)
{
	this->Width = Width;
	this->Height = Height;
	this->Origin = Point(0,0,0);
	this->Plan = Plane(
		Point(1,0.5,0.5),
		Point(1,0.5,-0.5),
		Point(1,-0.5,-0.5),
		Point(1,-0.5,0.5)
		);
}


__host__ __device__ Camera::~Camera(void)
{
}


__host__ __device__ Ray Camera::GetRay(int pixelX, int pixelY)
{
	float x = 1.0 - (float)pixelX / (float)Width;
	float y = 1.0 - (float)pixelY / (float)Height;

	return Ray(this->Origin, this->Plan.FaceToWorld(x, y));
}