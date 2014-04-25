//Dernière mise à jour le 09/01/2011

#ifndef __OBJ_LIB__
#define __OBJ_LIB__

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>

#include <SDL/SDL.h>
#include <GL/gl.h>
#include <GL/glu.h>

#include "sdlglutils.h"

std::string doubleSlash(std::string s);
std::string remplacerSlash(std::string s);
std::vector<std::string> splitSpace(std::string s);
std::string get_directory(std::string s);
float* vector2float(std::vector<float>& tableau);

class FloatVector
{
    /*
    Classe FloatVector : simple vecteur XYZ ou XYZA (dans le cas de couleurs).
    */
    public:
        FloatVector(float px=0,float py=0,float pz=0,float pa=0);
        /* FloatVector(float px=0,float py=0,float pz=0,float pa=0);
           Constructeur, prend en paramètres des flottants correspondant respectivement à x, y, z et a.
        */
        ~FloatVector();
        /* ~FloatVector();
           Destructeur, totalement inutile.
        */
        FloatVector operator=(const FloatVector &fv);
        /* FloatVector operator=(const FloatVector &fv);
           Affecte au vecteur courant le contenu du vecteur passé en argument.
           Retourne le vecteur courant ainsi modifié.
        */
        float x,y,z,a;
};

class Material
{
    /*
    Classe Material : définition d'un matériau, composé d'une couleur et d'un nom spécifique.
    */
    public:
        Material(float r,float g,float b,std::string n);
        /* Material(float r,float g,float b,std::string n);
           Constructeur, les trois premiers arguments représentent la couleur RGB du matériau et n est son nom.
        */
        Material(Material *mat);
        /* Material(Material *mat);
           Constructeur alternatif, affecte au matériau courant le contenu du matériau passé en argument.
        */
        ~Material();
        /* ~Material();
           Destructeur, totalement inutile.
        */

        FloatVector coul;
        std::string name;
};

class MeshObj
{
    /*
    Classe MeshObj : définition d'un modèle statique.
    */
    public:
        MeshObj(std::string,MeshObj *first=NULL);
        /* MeshObj(std::string,MeshObj *first=NULL);
           Constructeur, prend en arguments le nom du modèle à charger et le pointeur de la première frame si le modèle appartient à une animation (sinon laissez-le à NULL).
        */
        ~MeshObj();
        /* ~MeshObj();
           Destructeur, libère toute la mémoire qui lui a été allouée.
        */
        void charger_obj(std::string,MeshObj *first=NULL);
        /* void charger_obj(std::string,MeshObj *first=NULL);
           Charge un fichier OBJ et son MTL, prend en arguments le nom du modèle à charger et le pointeur de la première frame si le modèle appartient à une animation (sinon laissez-le à NULL). Cette fonction est appelée par le constructeur.
           Aucune valeur de retour.
        */
        void charger_mtl(std::string);
        /* void charger_mtl(std::string);
           Charge un fichier MTL, prend en argument le nom du fichier à charger. Cette fonction est appelée par charger_obj.
           Aucune valeur de retour.
        */
        void draw_model(bool nor=true,bool tex=false);
        /* void draw_model(bool nor=true,bool tex=false);
           Dessine le modèle, prend en arguments deux booléens représentant respectivement les normales et la texture. Si nor vaut true alors on prend en compte les normales, et si tex vaut true alors on applique la texture.
           Aucune valeur de retour.
        */
        void setMaterialsAndTex(std::vector<Material*> mats,GLuint tex);
        /* void setMaterialsAndTex(std::vector<Material*> mats,GLuint tex);
           Définit directement les matériaux et la texture du modèle, prend en arguments un vector<Material*> et la texture. Cette fonction est appelée par giveMaterialsAndTex.
           Aucune valeur de retour.
        */
        void giveMaterialsAndTex(MeshObj *target);
        /* void giveMaterialsAndTex(MeshObj *target);
           Modifie les matériaux et la texture de target en les remplaçant par ses propres matériaux et sa texture. Cette fonction est appelée par charger_obj uniquement lorsque first!=NULL.
           Aucune valeur de retour.
        */
    private:
        GLuint texture;
        int n_data;
        float *vertice,*normals,*textures,*colours;

        std::vector<Material*> materiaux;
};

class AnimMesh
{
    /*
    Classe AnimMesh : animation, contient chaque frame sous forme de modèle statique.
    */
    public:
        AnimMesh(int anframes,std::string name);
        /* AnimMesh(int anframes,std::string name);
           Constructeur, prend pour arguments le nombre de frames et le nom d'une frame sans son numéro et sans son extension (par exemple si c'est "personnage_000001.obj" on ne mettra que "personnage").
        */
        ~AnimMesh();
        /* ~AnimMesh();
           Destructeur, libère toutes les frames.
        */
        void draw(int fr,bool nor=true,bool tex=false);
        /* void draw(int fr,bool nor=true,bool tex=false);
           Dessine la frame fr, les deux autres arguments sont les mêmes que ceux de la fonction MeshObj::draw_model.
           Aucune valeur de retour.
        */
    private:
        int nframes;
        std::vector<MeshObj*> frames;
};

class VirtualAnim
{
    /*
    Class VirtualAnim : animation virtuelle stockée uniquement sous la forme d'entiers.
    */
    public:
        VirtualAnim();
        /* VirtualAnim();
           Constructeur, met les attributs à une valeur par défaut.
        */
        ~VirtualAnim();
        /* ~VirtualAnim();
           Destructeur, totalement inutile.
        */
        void start(int ab,int ae,int afps=25,bool back2beginning=false,int an_cycles=-1);
        /* void start(int ab,int ae,int afps=25,bool back2beginning=false,int an_cycles=-1);
           Démarre l'animation, prend pour arguments respectivement la frame de départ, la frame de fin, le nombre de frames par seconde, un booléen indiquant si une fois arrivé à la frame de fin on retourne à la frame 0 ou à la frame de début, et enfin le nombre de fois que cette animation doit être exécutée (-1 pour jouer l'animation en boucle).
           Aucune valeur de retour.
        */
        void stop();
        /* void stop();
           Arrête l'animation (la met en pause).
           Aucune valeur de retour.
        */
        void draw(AnimMesh *mesh,bool nor=true,bool tex=false);
        /* void draw(AnimMesh *mesh,bool nor=true,bool tex=false);
           Dessine la frame de mesh correspondant à la frame courante, les deux derniers arguments étant les mêmes que ceux de MeshObj::draw_model.
           Aucune valeur de retour.
        */
    private:
        int b,e,fps,fr,n_cycles,cycles;
        unsigned int tps,inter;
        bool back,playing;
};

#endif
