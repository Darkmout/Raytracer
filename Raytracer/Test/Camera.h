#pragma once
#include "device_launch_parameters.h"

#include "Point.h"
#include "Plane.h"
#include "Ray.h"

class Camera
{
public:
	int Width, Height;
	Vec3 Origin;
	Plane Plan;

	__host__ __device__ Camera(void)
	{
		this->Origin = Vec3(0,0,0);
		this->Plan = Plane(
			Vec3(1,0.5,0.5),
			Vec3(1,0.5,-0.5),
			Vec3(1,-0.5,-0.5),
			Vec3(1,-0.5,0.5)
			);
	}
	__host__ __device__ Camera(int Width, int Height)
	{
		this->Width = Width;
		this->Height = Height;
		this->Origin = Vec3(0,0,0);
		this->Plan = Plane(
			Vec3(1,0.5,0.5),
			Vec3(1,0.5,-0.5),
			Vec3(1,-0.5,-0.5),
			Vec3(1,-0.5,0.5)
			);
	}


	__host__ __device__ ~Camera(void)
	{}

	__host__ __device__ Ray GetRay(int pixelX, int pixelY)
	{
		float x = (float)pixelX / (float)Width;
		float y = 1.0-(float)pixelY / (float)Height;

		Ray ray = Ray(this->Origin, (this->Plan.FaceToWorld(x, y) - this->Origin));
		ray.Direction.Normalize();
		return ray;
	}
};

