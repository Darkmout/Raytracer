#ifndef _CAMERA_H
#define _CAMERA_H


#include "device_launch_parameters.h"


#include "Plane.cuh"
#include "Ray.cuh"
#include "Matrix3.cuh"
#include <stdio.h>
#include <math.h>

class Camera
{
public:

	Vec3 Origin;
	Plane Plan;

	__host__ __device__ Camera()
	{
		this->ToFront();
	}

	__host__ __device__ void ToOrigin()
	{
		this->Origin = Vec3(0,0,0);
		this->Plan = Plane(
			Vec3(1,-1,-1),
			Vec3(-1,-1,-1),
			Vec3(-1,1,-1),
			Vec3(1,1,-1)
			);
	}

	__host__ __device__ void ToUp()
	{
		this->Origin = Vec3(0,0,10);
		this->Plan = Plane(
			Vec3(1,-1,9),
			Vec3(-1,-1,9),
			Vec3(-1,1,9),
			Vec3(1,1,9)
			);
	}

	__host__ __device__ void ToFront()
	{
		this->Origin = Vec3(0,-10,0);
		this->Plan = Plane(
			Vec3(-1,-9,-1),
			Vec3(1,-9,-1),
			Vec3(1,-9,1),
			Vec3(-1,-9,1)
			);
	}



	__host__ __device__ ~Camera(void)
	{}

	__host__ __device__ Ray GetRay(int pixelX, int pixelY, int Width, int Height)
	{
		float x = (float)pixelX / (float)Width;
		float y = (float)pixelY / (float)Height;

		Ray ray = Ray(this->Origin, (this->Plan.FaceToWorld(x, y) - this->Origin));
		ray.Direction.Normalize();
		return ray;
	}

	__host__ __device__ void Move(Vec3 MoveLocal)
	{
		//CheckSize();
		// MoveLocal.Cross(Plan.Normal);
		Vec3 v0v1 = Plan.v1 - Plan.v0;
		Vec3 MoveGlobal = Vec3(
			Plan.Normal.x * MoveLocal.x  + v0v1.x * MoveLocal.y,
			Plan.Normal.y * MoveLocal.x  + v0v1.y * MoveLocal.y,
			Plan.Normal.z * MoveLocal.x  + v0v1.z * MoveLocal.y + MoveLocal.z);

		this->Origin = this->Origin + MoveGlobal;
		this->Plan = this->Plan + MoveGlobal;
	}

	__host__ __device__ void Rotation(int x, int y)
	{
		Matrix3 rotation((float)((float)y)*(0.01f), 0, (float)((float)x)*(0.01f));
	//	this->Plan = this->Plan - this->Origin;

		this->Plan.v0 = this->Plan.v0 * rotation;
		this->Plan.v1 = this->Plan.v1 * rotation;
		this->Plan.v2 = this->Plan.v2 * rotation;
		this->Plan.v3 = this->Plan.v3 * rotation;
		this->Origin = this->Origin * rotation;



		/*
		float gamma = (float)((float)x)*(0.01f); 
		float beta = (float)((float)y)*(0.01f); 

		//move the plane to rotate at the origin;
		this->Plan = this->Plan - this->Origin;

		//applique the two rotation matrices  http://fr.wikipedia.org/wiki/Matrice_de_rotation#Les_matrices_de_base
		// and http://blog.wolfire.com/2010/07/Linear-algebra-for-game-developers-part-4

		//rotation in function of the Z-axis
		Matrix3 rotateZ = Matrix3(ZROTATION, gamma);

		this->Plan.v0 = this->Plan.v0 * rotateZ;
		this->Plan.v1 = this->Plan.v1 * rotateZ;
		this->Plan.v2 = this->Plan.v2 * rotateZ;
		this->Plan.v3 = this->Plan.v3 * rotateZ;

		//rotation in function of the axis from v0->v1 of the camera plan
		Vec3 Xaxis = this->Plan.v3 - this->Plan.v0;
		Vec3 Yaxis = (this->Plan.v3 - this->Plan.v0).Cross(this->Plan.v1 - this->Plan.v0);
		Vec3 Zaxis = this->Plan.v0 - this->Plan.v1;
		Xaxis.Normalize();
		Yaxis.Normalize();
		Zaxis.Normalize();
		Matrix3 NewCoordinate = Matrix3(
			Xaxis.x, Xaxis.y, Xaxis.z,
			Yaxis.x, Yaxis.y, Yaxis.z,
			Zaxis.x, Zaxis.y, Zaxis.z);

		Matrix3 Inverse = NewCoordinate.Inverse();

		this->Plan.v0 = this->Plan.v0 * Inverse;
		this->Plan.v1 = this->Plan.v1 * Inverse;
		this->Plan.v2 = this->Plan.v2 * Inverse;
		this->Plan.v3 = this->Plan.v3 * Inverse;

		Matrix3 rotateX = Matrix3(ZROTATION, beta);
		this->Plan.v0 = this->Plan.v0 * rotateX;
		this->Plan.v1 = this->Plan.v1 * rotateX;
		this->Plan.v2 = this->Plan.v2 * rotateX;
		this->Plan.v3 = this->Plan.v3 * rotateX;
		
		this->Plan.v0 = this->Plan.v0 * NewCoordinate;
		this->Plan.v1 = this->Plan.v1 * NewCoordinate;
		this->Plan.v2 = this->Plan.v2 * NewCoordinate;
		this->Plan.v3 = this->Plan.v3 * NewCoordinate;

		*/

		/*
		this->Plan.v0.x = this->Plan.v0.x * cos(gamma) - this->Plan.v0.y * sin(gamma);
		this->Plan.v0.y = this->Plan.v0.x * sin(gamma) + this->Plan.v0.y * cos(gamma);

		this->Plan.v1.x = this->Plan.v1.x * cos(gamma) - this->Plan.v1.y * sin(gamma);
		this->Plan.v1.y = this->Plan.v1.x * sin(gamma) + this->Plan.v1.y * cos(gamma);

		this->Plan.v2.x = this->Plan.v2.x * cos(gamma) - this->Plan.v2.y * sin(gamma);
		this->Plan.v2.y = this->Plan.v2.x * sin(gamma) + this->Plan.v2.y * cos(gamma);

		this->Plan.v3.x = this->Plan.v3.x * cos(gamma) - this->Plan.v3.y * sin(gamma);
		this->Plan.v3.y = this->Plan.v3.x * sin(gamma) + this->Plan.v3.y * cos(gamma);
		*/
		/*
		beta = (float)((float)y)*(0.01f) * this->Plan.Normal.Dot(Vec3(1,0,0)); 
		//rotation in function of the y-axis
		this->Plan.v0.x = this->Plan.v0.x * cos(beta) + this->Plan.v0.z * sin(beta);
		this->Plan.v0.z = -this->Plan.v0.x * sin(beta) + this->Plan.v0.z* cos(beta);

		this->Plan.v1.x = this->Plan.v1.x * cos(beta) + this->Plan.v1.z * sin(beta);
		this->Plan.v1.z = -this->Plan.v1.x * sin(beta) + this->Plan.v1.z* cos(beta);

		this->Plan.v2.x = this->Plan.v2.x * cos(beta) + this->Plan.v2.z * sin(beta);
		this->Plan.v2.z = -this->Plan.v2.x * sin(beta) + this->Plan.v2.z* cos(beta);

		this->Plan.v3.x = this->Plan.v3.x * cos(beta) + this->Plan.v3.z * sin(beta);
		this->Plan.v3.z = -this->Plan.v3.x * sin(beta) + this->Plan.v3.z* cos(beta);

		beta = (float)((float)y)*(0.01f) * this->Plan.Normal.Dot(Vec3(0,1,0)); 
		//rotation in function of the x-axis
		this->Plan.v0.y = this->Plan.v0.y * cos(beta) - this->Plan.v0.z * sin(beta);
		this->Plan.v0.z = this->Plan.v0.y * sin(beta) + this->Plan.v0.z * cos(beta);

		this->Plan.v1.y = this->Plan.v1.y * cos(beta) - this->Plan.v1.z * sin(beta);
		this->Plan.v1.z = this->Plan.v1.y * sin(beta) + this->Plan.v1.z * cos(beta);

		this->Plan.v2.y = this->Plan.v2.y * cos(beta) - this->Plan.v2.z * sin(beta);
		this->Plan.v2.z = this->Plan.v2.y * sin(beta) + this->Plan.v2.z * cos(beta);

		this->Plan.v3.y = this->Plan.v3.y * cos(beta) - this->Plan.v3.z * sin(beta);
		this->Plan.v3.z = this->Plan.v3.y * sin(beta) + this->Plan.v3.z * cos(beta);

		//move back to the position after the rotation
		*/
	//	this->Plan = this->Plan + this->Origin;
		this->Plan.ActualisePosition();

		printf("%f\n", (this->Plan.v0 - this->Plan.v1).Length());

	}
};

#endif /* CAMERA_H*/