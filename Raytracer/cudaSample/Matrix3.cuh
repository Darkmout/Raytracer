#pragma once
#include "device_launch_parameters.h"
#include "cuda_runtime.h"

#define ZROTATION 1
#define XROTATION 2
#define YROTATION 3

class Matrix3
{
public:
	float aa, ab, ac, ba, bb, bc, ca, cb, cc;

	__host__ __device__ Matrix3(void)
	{
		this->aa = 0;
		this->ab = 0;
		this->ac = 0;
		this->ba = 0;
		this->bb = 0;
		this->bc = 0;
		this->ca = 0;
		this->cb = 0;
		this->cc = 0;
	}
	__host__ __device__ Matrix3(float aa, float ab, float ac, float ba, float bb, float bc, float ca, float cb, float cc)
	{
		this->aa = aa;
		this->ab = ab;
		this->ac = ac;
		this->ba = ba;
		this->bb = bb;
		this->bc = bc;
		this->ca = ca;
		this->cb = cb;
		this->cc = cc;
	}
	__host__ __device__ Matrix3(int type, float angle)
	{
		if(type == ZROTATION)
		{
			this->aa = cos(angle);
			this->ab = sin(angle);
			this->ac = 0;
			this->ba = -sin(angle);
			this->bb = cos(angle);
			this->bc = 0;
			this->ca = 0;
			this->cb = 0;
			this->cc = 1;
		}
		else if(type == YROTATION)
		{
			this->aa = cos(angle);
			this->ab = 0;
			this->ac = -sin(angle);
			this->ba = 0;
			this->bb = 1;
			this->bc = 0;
			this->ca = sin(angle);
			this->cb = 0;
			this->cc = cos(angle);
		}
		else if(type == XROTATION)
		{
			this->aa = 1;
			this->ab = 0;
			this->ac = 0;
			this->ba = 0;
			this->bb = cos(angle);
			this->bc = sin(angle);
			this->ca = 0;
			this->cb = -sin(angle);
			this->cc = cos(angle);
		}
	}
	//create rotation matrix from euler angles
	__host__ __device__ Matrix3(float x, float y, float z)
	{
		this->aa = cos(y)*cos(z);
		this->ab = cos(y)*sin(z);
		this->ac = -sin(y);
		this->ba = sin(x)*sin(y)*cos(z) - cos(x)*sin(z);
		this->bb = sin(x)*sin(y)*sin(z) + cos(x)*cos(z);
		this->bc = sin(x)*cos(y);
		this->ca = cos(x)*sin(y)*cos(z) + sin(x)*sin(z);
		this->cb = cos(x)*sin(y)*sin(z) - sin(x)*cos(z);
		this->cc = cos(x)*cos(y);
	}


	__host__ __device__ Matrix3 Traspose()
	{
		return Matrix3(aa, ba, ca, ab, bb, cb, ac, bc, cc);
	}

	__host__ __device__ float Determinant()
	{
		return aa * (bb * cc - cb * bc) - ba * (ab * cc - cb * ac) + ca * (ab * bc - bb * ac);
	}

	//return M^-1
	//http://www.mathsisfun.com/algebra/matrix-inverse-minors-cofactors-adjugate.html
	__host__ __device__ Matrix3 Inverse()
	{

		Matrix3 Minors = Matrix3(
			bb * cc - cb * bc, ba * cc - ca * bc, ba * cb - ca * bb,
			ab * cc - cb * ac, aa * cc - ca * ac, aa * cb - ca * ab,
			ab * bc - bb * ac, aa * bc - ba * ac, aa * bb - ba * ab);

		//matrix of cofactrs
		Minors.ab *= -1;
		Minors.ba *= -1;
		Minors.bc *= -1;
		Minors.cb *= -1;

		//matrix of Adjugate
		Matrix3 Adjugate = Minors.Traspose();

		return Adjugate * (1 / this->Determinant());
	}

	__host__ __device__  Matrix3 operator*(float o){
		return Matrix3(
			aa * o, ab * o, ac * o,
			ba * o, bb * o, bc * o,
			ca * o, cb * o, cc * o); 
	}

};

