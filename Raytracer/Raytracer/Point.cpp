#include "Point.h"


Point::Point(void)
{
	this->x = 0;
	this->y = 0;
	this->z = 0;
}

Point::Point(float x, float y, float z)
{
	this->x = x;
	this->y = y;
	this->z = z;
}

Point::~Point(void)
{
}

Vector Point::To(Point Direction)
{
	Point result = (Direction - *this );
	return Vector(result.x, result.y, result.z);
}