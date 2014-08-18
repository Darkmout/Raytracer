#ifndef _PLANE_H
#define _PLANE_H

#define FLOAT_MAX 999999999.9

#include "Vec3.cuh"
#include "Ray.cuh"
#include "device_launch_parameters.h"
#include "cuda.h"
#include "Material.cuh"

class Plane
{
public:
	Vec3 v0, v1, v2, v3, Normal, vt0, vt1, vt2, vt3;
	Material* material;
	float D;

	__host__ __device__ Plane()
	{}

	//the points have to be anti-clockwise
	__host__ __device__ Plane(Vec3 v0, Vec3 v1, Vec3 v2, Vec3 v3)
	{
		this->v0 = v0;
		this->v1 = v1;
		this->v2 = v2;
		this->v3 = v3;

		this->ActualisePosition();

	}

	__host__ __device__ Plane(Vec3 v0, Vec3 v1, Vec3 v2, Vec3 v3, Vec3 vt0, Vec3 vt1, Vec3 vt2, Vec3 vt3, Material *material)
	{
		this->v0 = v0;
		this->v1 = v1;
		this->v2 = v2;
		this->v3 = v3;
		this->vt0 = vt0;
		this->vt1 = vt1;
		this->vt2 = vt2;
		this->vt3 = vt3;
		this->material = material;

		this->ActualisePosition();
	}


	__host__ __device__ void ActualisePosition()
	{

		this->Normal = (v1-v0).Cross(v3-v0);
		this->Normal.Normalize();

		this->D = Normal.Dot(v0);
		/*
		this->v0v1 = v1 - v0;
		this->v1v2 = v2 - v1;
		this->v2v3 = v3 - v2;
		this->v3v0 = v0 - v3;
		*/
	}

	__host__ __device__~Plane(void)
	{}

	__host__ __device__ float Intersect(Ray IncidentRay)
	{
		//http://www.scratchapixel.com/lessons/3d-basic-lessons/lesson-9-ray-triangle-intersection/ray-triangle-intersection-geometric-solution/

		float NormalDotRayDirection = Normal.Dot(IncidentRay.Direction);
		//if the ray and the plane are parallel or in the same direction
		if(NormalDotRayDirection < 0)
			return FLOAT_MAX;

		//compute the distance t from the place
		float t = - ( Normal.Dot(IncidentRay.Origin) + D) / (NormalDotRayDirection);
		//if the plane is behind the origin of the ray
		if(t<0.1)
			return FLOAT_MAX;

		//compute the point on the plane
		Vec3 P = IncidentRay.Origin + (IncidentRay.Direction * t); 

		Vec3 C0 = P - v0;
		Vec3 C1 = P - v1;
		Vec3 C2 = P - v2;
		Vec3 C3 = P - v3;


		if(	Normal.Dot((v1-v0).Cross(C0))>0 &&
			Normal.Dot((v2-v1).Cross(C1))>0 &&
			Normal.Dot((v3-v2).Cross(C2))>0 &&
			Normal.Dot((v0-v3).Cross(C3))>0)
			return t;
		else
			return FLOAT_MAX;
	}

	__host__ __device__ Vec3 FaceToWorld(float x, float y) //TODO: doesn't seem to work
	{
		Vec3 X = ((v1 - v0) * x);
		Vec3 Y = ((v3 - v0) * y);

		Vec3 result = v0 + (X + Y); 
		return result;
	}

	//__host__ __device__ Vec3 WorldToFace(Vec3 Point)
	//{
	///*	float x = 
	//	Vec3 result = Vec3(
	//		this->v0.x*x*y + this->v1.x*(1-x)*(y) + this->v2.x*(x)*(1-y) + this->v3.x*(1-x)*(1-y),
	//		this->v0.y*x*y + this->v1.y*(1-x)*(y) + this->v2.y*(x)*(1-y) + this->v3.y*(1-x)*(1-y),
	//		this->v0.z*x*y + this->v1.z*(1-x)*(y) + this->v2.z*(x)*(1-y) + this->v3.z*(1-x)*(1-y)
	//		);

	//	return result;*/
	//	//TODO
	//	return nill;
	//}

	__host__ __device__  Plane operator-(Vec3 o){

		return Plane(this->v0 - o, this->v1 - o, this->v2 - o, this->v3 - o);
	}

	__host__ __device__  Plane operator+(Vec3 o){

		return Plane(this->v0 + o, this->v1 + o, this->v2 + o, this->v3 + o);
	}
};

#endif /*_PLANE_H */