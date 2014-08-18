#ifndef _MODEL_H
#define _MODEL_H

#include "Material.cuh"
#include "Vec3.cuh"
#include "Plane.cuh"
#include "Camera.cuh"


#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <map>

#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <helper_cuda.h>


using namespace std;

class Model
{

public:
	std::vector<Plane> Planes;
	Plane *d_scene;
	Camera camera;
	std::map<std::string, Material*> Materials;

	//default constructor
	Model(){}

	Model(string filePath)
	{
		float x, y, z;
		int v0, v1, v2, v3;
		int vt0, vt1, vt2, vt3;
		char charLol;
		ifstream infile(filePath);
		string line;
		int lineNumber = 0;
		vector<Vec3> Sommets;
		vector<Vec3> vt;
		vt.push_back(Vec3());
		Sommets.push_back(Vec3());//to set index at 1
		Material* currentMaterial;

		try{
			while (getline(infile, line))
			{
				lineNumber++;

				if(line[0] == '#') //commentaire
					continue;

				if(line[0] == 'o') //object name
					continue;

				if(line[0] == 'v' && line[1] == 't') //texture "vt float float"
				{
					istringstream iss(line);
					if (!(iss >> charLol >> charLol >> x >> y)) { 
						throw new exception(); 
					} 

					vt.push_back(Vec3(x,y,0));

					continue;
				}

				if(line[0] == 'v' && line[1] == 'n') //normal "vn float float float"
					continue;

				if(line[0] == 'v') //Vertex "v float float float"
				{
					istringstream iss(line);
					if (!(iss >> charLol >> x >> y >> z)) { 
						throw new exception(); 
					} 

					Sommets.push_back(Vec3(x,y,z));

					continue;
				}

				if(line[0] == 'f') //face "f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3"
				{
					istringstream iss(line);
					if (!(iss >> charLol >> v0 >> charLol >> vt0 >> v1 >> charLol >> vt1  >> v2 >> charLol >> vt2  >> v3 >> charLol >> vt3 )) { 
						throw new exception(); 
					}

					this->Planes.push_back(Plane(Sommets[v0], Sommets[v1], Sommets[v2], Sommets[v3], vt[vt0], vt[vt1], vt[vt2], vt[vt3], currentMaterial));

					continue;
				}

				if(line.find("mtllib") != string::npos)
				{
					string mtllibName = line.substr(7);
					Loadmtllib(filePath.substr(0, filePath.find_last_of("/\\")) + "\\" + mtllibName); //TODO: linux paths 
					continue;
				}

				if(line.find("usemtl") != string::npos)
				{
					currentMaterial = Materials.at(line.substr(7));
				}

			}
		}
		catch( ... )
		{
			printf("error in file %s line %d", filePath.c_str(), lineNumber);
			exit(1);
		}

		checkCudaErrors(cudaMalloc(&d_scene,   sizeof(Plane) * Planes.size()));
		UpdateScene();

	}

	~Model(void)
	{
		//checkCudaErrors(cudaFree(d_scene));
	}

	void Loadmtllib(std::string path)
	{
		ifstream infile(path);
		string line;
		int lineNumber = 0;
		Material* currentMaterial;

		try{
			while (getline(infile, line))
			{
				lineNumber++;

				if(int pos = line.find("newmtl") != string::npos)
				{
					Material *d_Material;						
					checkCudaErrors(cudaMalloc(&d_Material, sizeof(Material)));

					currentMaterial = d_Material;
					Materials.emplace(line.substr(7), d_Material);
				}
				else if(line[0] == 'K' && line[1] == 'd') //texture "vt float float"
				{
					istringstream iss(line);
					char space;
					float R, G, B;
					if (!(iss >> space >> space >> R >> G >> B)) { 
						throw new exception(); 
					} 
					Material host = Material(make_uchar4(R*255, G*255,B*255, 0));
					checkCudaErrors(cudaMemcpy(currentMaterial, &host,sizeof(Material), cudaMemcpyHostToDevice));
				}
				else if(int pos = line.find("map_Kd") != string::npos)
				{
					ifstream texture;
					string textureName = line.substr(7);
					texture.open(path.substr(0, path.find_last_of("/\\")) + "\\" + textureName); //TODO: linux paths 
				}
			}
		}
		catch( ... )
		{
			printf("error in file %s line %d", path.c_str(), lineNumber);
			exit(1);
		}

	}

	void LoadHdr(std::string path)
	{
			ifstream infile(path);
	}

	void UpdateScene()
	{
		if(Planes.size())
			checkCudaErrors(cudaMemcpy(d_scene, &Planes[0], sizeof(Plane) * Planes.size(), cudaMemcpyHostToDevice));
	}
};

#endif
