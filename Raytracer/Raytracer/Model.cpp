#include "Model.h"
#include <fstream>
#include <vector>

using namespace std;

//static
void Model::LoadObj(string path)
{
	ifstream file(path.c_str());
	string line;
	vector<Point> vertices;

	if(file)
    {
        while(getline(file,line))
        {
			if(line[0] == 'v' && line[1] != 't')
			{
				//scanf("%d %d %d", line);
				//vertices.push_back(new Vertex(0,0,0));
			}
		}
	}

}

Model::Model(void)
{
}


Model::~Model(void)
{
}

