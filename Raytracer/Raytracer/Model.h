#pragma once
#include <string>

typedef struct Point
{
	float x, y, z;
} Point;

typedef struct Vector
{
	Point origin;
	Point direction;
} Vector;

typedef struct Face
{
	Point A,B,C,D;
	Vector normal;
	int  a,b,c,d;

} Face;

typedef struct Camera
{
	Point origin;
	Face plan;
} Camera;


class Model
{
public:
	static void LoadObj(std::string);
	Model(void);
	~Model(void);
};


