#pragma once

#include "Model.h"
#include "Vec3.h"
#include "Plane.h"
#include "Camera.h"

#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <list>

#include "device_launch_parameters.h"
#include "cuda_runtime.h"


class Model
{
public:
	std::vector<Plane> Planes;
	Plane *d_scene;
	Camera camera;

	Model(std::string);
	Model(void);
	~Model(void);

	void UpdateScene();
};


