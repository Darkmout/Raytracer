#include "Ray.h"


Ray::Ray(void)
{
}

Ray::Ray(Point Origin, Point Direction)
{
	this->Origin = Origin;
	this->Direction = Origin.ToVector(Direction);
}

Ray::Ray(Point Origin, Vec3 Direction)
{
	this->Origin = Origin;
	this->Direction = Direction;
}


Ray::~Ray(void)
{
}
