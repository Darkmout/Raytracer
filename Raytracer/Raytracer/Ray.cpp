#include "Ray.h"


Ray::Ray(void)
{
}

Ray::Ray(Point Origin, Point Direction)
{
	this->Origin = Origin;
	this->Direction = Origin.To(Direction);
}

Ray::Ray(Point Origin, Vector Direction)
{
	this->Origin = Origin;
	this->Direction = Direction;
}


Ray::~Ray(void)
{
}
