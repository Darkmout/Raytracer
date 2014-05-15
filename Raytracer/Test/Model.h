#pragma once

#include "Model.h"
#include "Vec3.h"
#include "Plane.h"

#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <list>

#include "device_launch_parameters.h"

class Model
{
public:
	std::vector<Plane> Planes;
	Model(std::string);
	Model(void);
	~Model(void);
};


