#include <GL/gl.h>
#include <GL/glu.h>

#include "objlib.h"

using namespace std;

string doubleSlash(string s)
{
    //Remplace "//" par "/1/".
    string s1="";
    for(unsigned int i=0;i<s.size();i++)
    {
        if(i<s.size()-1&&s[i]=='/'&&s[i+1]=='/')
        {
            s1+="/1/";
            i++;
        }
        else
            s1+=s[i];
    }
    return s1;
}
string remplacerSlash(string s)
{
    //Remplace les '/' par des espaces.
    string ret="";
    for(unsigned int i=0;i<s.size();i++)
    {
        if(s[i]=='/')
            ret+=' ';
        else
            ret+=s[i];
    }
    return ret;
}
vector<string> splitSpace(string s)
{
    //Eclate une chaîne au niveau de ses espaces.
    vector<string> ret;
    string s1="";
    for(unsigned int i=0;i<s.size();i++)
    {
        if(s[i]==' '||i==s.size()-1)
        {
            if(i==s.size()-1)
                s1+=s[i];
            ret.push_back(s1);
            s1="";
        }
        else
            s1+=s[i];
    }
    return ret;
}
string get_directory(string s)
{
    string s1="",s2="";
    for(unsigned int i=0;i<s.size();i++)
    {
        if(s[i]=='/'||s[i]=='\\')
        {
            s1+=s2+"/";
            s2="";
        }
        else
            s2+=s[i];
    }
    return s1;
}
float* vector2float(vector<float>& tableau)
{
    float* t=NULL;
    t=(float*)malloc(tableau.size()*sizeof(float));
    if(t==NULL||tableau.empty())
    {
        float *t1=(float*)malloc(sizeof(float)*3);
        for(int i=0;i<3;i++)
            t1[i]=0.;
        return t1;
    }

    for(unsigned int i=0;i<tableau.size();i++)
        t[i]=tableau[i];
    return t;
}

FloatVector::FloatVector(float px,float py,float pz,float pa):x(px),y(py),z(pz),a(pa)
{
}
FloatVector::~FloatVector()
{
}
FloatVector FloatVector::operator=(const FloatVector &fv)
{
    x=fv.x;
    y=fv.y;
    z=fv.z;
    a=fv.a;

    return *this;
}

Material::Material(float r,float g,float b,string n):name(n)
{
    coul.x=r;
    coul.y=g;
    coul.z=b;
}
Material::Material(Material *mat)
{
    coul=mat->coul;
    name=mat->name;
}
Material::~Material()
{
}

MeshObj::MeshObj(string s,MeshObj *first)
{
    charger_obj(s,first);
}
MeshObj::~MeshObj()
{
    free(vertice);
    free(normals);
    free(textures);
    free(colours);

    for(unsigned int i=0;i<materiaux.size();i++)
        delete materiaux[i];
    materiaux.clear();
}
void MeshObj::charger_obj(string nom,MeshObj *first)
{
    vector<FloatVector> ver,nor,tex,col;
    std::vector<unsigned int> iv,it,in;

    ifstream fichier(nom.c_str(),ios::in);

    string ligne,curname="";

    if(first) //Si ce n'est pas la première frame, on demande à celle-ci de nous passer ses matériaux et sa texture (et elle accepte ^^ )
        first->giveMaterialsAndTex(this);

    if(fichier)
    {
        while(getline(fichier,ligne))
        {
            if(ligne[0]=='v') //Coordonnées de points (vertex, texture et normale)
            {
                if(ligne[1]==' ') //Vertex
                {
                    char x[255],y[255],z[255];
                    sscanf(ligne.c_str(),"v %s %s %s",x,y,z);
                    ver.push_back(FloatVector(strtod(x,NULL),strtod(y,NULL),strtod(z,NULL)));
                }
                else if(ligne[1]=='t') //Texture
                {
                    char x[255],y[255];
                    sscanf(ligne.c_str(),"vt %s %s",x,y);
                    tex.push_back(FloatVector(strtod(x,NULL),strtod(y,NULL)));
                }
                else if(ligne[1]=='n') //Normale
                {
                    char x[255],y[255],z[255];
                    sscanf(ligne.c_str(),"vn %s %s %s",x,y,z);
                    nor.push_back(FloatVector(strtod(x,NULL),strtod(y,NULL),strtod(z,NULL)));
                }
            }
            else if(ligne[0]=='f') //Indice faces
            {
                ligne=doubleSlash(ligne); //On remplace "//" par "/1/" dans toute la ligne
                ligne=remplacerSlash(ligne); //On remplace les '/' par des espaces, ex : pour "f 1/2/3 4/5/6 7/8/9" on obtiendra "f 1 2 3 4 5 6 7 8 9"

                vector<string> termes=splitSpace(ligne.substr(2)); //On éclate la chaîne en ses espaces (le substr permet d'enlever "f ")

                int ndonnees=(int)termes.size()/3;
                for(int i=0;i<(ndonnees==3?3:4);i++) //On aurait très bien pu mettre i<ndonnees mais je veux vraiment limiter à 3 ou 4
                {
                    iv.push_back(strtol(termes[i*3].c_str(),NULL,10)-1);
                    it.push_back(strtol(termes[i*3+1].c_str(),NULL,10)-1);
                    in.push_back(strtol(termes[i*3+2].c_str(),NULL,10)-1);
                }
                if(ndonnees==3) //S'il n'y a que 3 sommets on duplique le dernier pour faire un quad ayant l'apparence d'un triangle
                {
                    iv.push_back(strtol(termes[0].c_str(),NULL,10)-1);
                    it.push_back(strtol(termes[1].c_str(),NULL,10)-1);
                    in.push_back(strtol(termes[2].c_str(),NULL,10)-1);
                }

                for(unsigned int i=0;i<materiaux.size();i++)
                    if(materiaux[i]->name==curname)
                    {
                        for(int j=0;j<4;j++)
                            col.push_back(materiaux[i]->coul); //On ajoute la couleur correspondante
                        break;
                    }
            }
            else if(ligne[0]=='m'&&first==NULL)//fichier MTL et si c'est la première frame (comme ça on ne charge pas plusieurs fois le même MTL et la même texture)
                charger_mtl(get_directory(nom)+ligne.substr(7));
            else if(ligne[0]=='u')//utiliser un MTL
                curname=ligne.substr(7);
        }
        fichier.close();

        vector<float> tv(0),tc(0),tn(0),tt(0);
        for(unsigned int i=0;i<iv.size();i++)
            if(iv[i]<ver.size())
            {
                tv.push_back(ver[iv[i]].x);
                tv.push_back(ver[iv[i]].y);
                tv.push_back(ver[iv[i]].z);

                tc.push_back(col[i].x);
                tc.push_back(col[i].y);
                tc.push_back(col[i].z);
                tc.push_back(col[i].a);
            }

        for(unsigned int i=0;i<in.size();i++)
            if(in[i]<nor.size())
            {
                tn.push_back(nor[in[i]].x);
                tn.push_back(nor[in[i]].y);
                tn.push_back(nor[in[i]].z);
            }

        for(unsigned int i=0;i<it.size();i++)
            if(it[i]<tex.size())
            {
                tt.push_back(tex[it[i]].x);
                tt.push_back(tex[it[i]].y);
            }

        vertice=vector2float(tv);
        normals=vector2float(tn);
        textures=vector2float(tt);
        colours=vector2float(tc);
        n_data=iv.size();
    }
    else
    {
        fprintf(stderr, "Le fichier %s n'a pas pu etre charge...",nom.c_str());
        exit(EXIT_FAILURE);
    }

    ver.clear();
    nor.clear();
    tex.clear();
    col.clear();

    iv.clear();
    it.clear();
    in.clear();
}
void MeshObj::charger_mtl(string nom)
{
    ifstream fichier(nom.c_str(),ios::in);
    string curname="";
    if(fichier)
    {
        string ligne="";
        while(getline(fichier,ligne))
        {
            if(ligne[0]=='n') //nouveau materiau
                curname=ligne.substr(7);
            else if(ligne[0]=='K'&&ligne[1]=='d') //couleur
            {
                vector<string> termes=splitSpace(ligne.substr(3));
                materiaux.push_back(new Material((float)strtod(termes[0].c_str(),NULL),(float)strtod(termes[1].c_str(),NULL),(float)strtod(termes[2].c_str(),NULL),curname));
            }
            else if(ligne[0]=='m'&&ligne[5]=='d')//map_Kd (texture)
            {
                string f=get_directory(nom)+ligne.substr(7);
                texture=loadTexture(f.c_str());
            }
            else if(ligne[0]=='d') //opacité
            {
                string n=ligne.substr(2);
                materiaux[materiaux.size()-1]->coul.a=strtod(n.c_str(),NULL);
            }
        }
        fichier.close();
    }
    else
    {
        fprintf(stderr,"Erreur lors de la lecture de %s...",nom.c_str());
        exit(EXIT_FAILURE);
    }
}
void MeshObj::draw_model(bool nor,bool tex)
{
    glEnableClientState(GL_VERTEX_ARRAY);
    if(nor)
        glEnableClientState(GL_NORMAL_ARRAY);
    if(tex)
    {
        glEnableClientState(GL_TEXTURE_COORD_ARRAY);
        glBindTexture(GL_TEXTURE_2D,texture);
    }
    glEnableClientState(GL_COLOR_ARRAY);

    glVertexPointer(3,GL_FLOAT,0,vertice);

    if(tex)
        glTexCoordPointer(2,GL_FLOAT,0,textures);
    if(nor)
        glNormalPointer(GL_FLOAT,0,normals);
    glColorPointer(4,GL_FLOAT,0,colours);

    glDrawArrays(GL_QUADS,0,n_data);

    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
}
void MeshObj::setMaterialsAndTex(std::vector<Material*> mats,GLuint tex)
{
    materiaux.clear();
    for(unsigned int i=0;i<mats.size();i++)
        materiaux.push_back(new Material(mats[i]));
    texture=tex;
}
void MeshObj::giveMaterialsAndTex(MeshObj *target)
{
    target->setMaterialsAndTex(materiaux,texture);
}

AnimMesh::AnimMesh(int anframes,std::string name):nframes(anframes)
{
    for(int i=1;i<nframes;i++)
    {
        string nom=name+string("_");
        char n[32]="";
        sprintf(n,"%d",i);
        nom+=string(n)+string(".obj");
        
        if(i>1)
            frames.push_back(new MeshObj(nom,frames[0]));
        else
            frames.push_back(new MeshObj(nom));
    }
}
AnimMesh::~AnimMesh()
{
    for(unsigned int i=0;i<frames.size();i++)
        delete frames[i];
    frames.clear();
}
void AnimMesh::draw(int fr,bool nor,bool tex)
{
    frames[fr-1]->draw_model(nor,tex);
}

VirtualAnim::VirtualAnim():b(0),e(0),fps(0),fr(0),n_cycles(0),cycles(0),tps(0),playing(false)
{
}
VirtualAnim::~VirtualAnim()
{
}
void VirtualAnim::start(int ab,int ae,int afps,bool back2beginning,int an_cycles)
{
    b=ab;
    e=ae;
    n_cycles=an_cycles;
    fps=afps;
    back=back2beginning;
    inter=1000/fps;

    fr=b;
    cycles=0;
    playing=true;
    tps=SDL_GetTicks();
}
void VirtualAnim::stop()
{
    playing=false;
}
void VirtualAnim::draw(AnimMesh *mesh,bool nor,bool tex)
{
    mesh->draw(fr+1,nor,tex);

    if(SDL_GetTicks()-tps>=inter&&playing)
    {
        fr++;
        if(fr>=e)
        {
            fr=(back?0:b);
            cycles++;
        }
        if(n_cycles!=-1&&cycles>=n_cycles)
            stop();
        tps=SDL_GetTicks();
    }
}
