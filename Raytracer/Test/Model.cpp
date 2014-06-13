#include "Model.h"
using namespace std;


Model::Model(string filePath)
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

				this->Planes.push_back(Plane(Sommets[v0], Sommets[v1], Sommets[v2], Sommets[v3], vt[vt0], vt[vt1], vt[vt2], vt[vt3]));

				continue;
			}

			if(line.find("mtllib") != string::npos)
			{
				string mtllibName = line.substr(7);
				Loadmtllib(filePath.substr(0, filePath.find_last_of("/\\")) + "\\" + mtllibName); //TODO: linux paths 
				continue;
			}

			//istringstream iss(line);
			//int a, b;
			//if (!(iss >> a >> b)) { break; } // error

			//// process pair (a,b)
		}
	}
	catch( ... )
	{
		printf("error in file input line %d", lineNumber);
		exit(0);
	}

	UpdateScene();

}

Model::Model(void)
{
}


Model::~Model(void)
{
}


void Model::Loadmtllib(string path)
{
	ifstream infile(path);
	string line;
	int lineNumber = 0;
	string material;

	try{
		while (getline(infile, line))
		{
			lineNumber++;
			
			if(int pos = line.find("newmtl") != string::npos)
			{
				Materials.mtls[line.substr(7)].Name = line.substr(7);
				material = line.substr(7);
			}
		}
	}
	catch( ... )
	{
		printf("error in file input line %d", lineNumber);
		exit(0);
	}

}

void Model::UpdateScene()
{


	cudaMalloc(&d_scene,   sizeof(Plane) * Planes.size());
	cudaMemcpy(d_scene, &Planes[0], sizeof(Plane) * Planes.size(), cudaMemcpyHostToDevice);

	camera = Camera();

}