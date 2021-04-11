#if 0
#include <iostream>
#include <vector>
#include "obj.h"
#include <Windows.h>
#include <GL\glut.h>
using namespace std;

void display(void) {
	glClear(GL_COLOR_BUFFER_BIT);
	glColor3f(1.0, 1.0, 1.0);

	cObj obj("media/dragon.obj");
	obj.render();


	glFlush();
}

int main(int argc, char** argv)
{
	//Initialise GLUT with command-line parameters. 
	glutInit(&argc, argv);

	//Set Display Mode
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);

	//Set the window size
	glutInitWindowSize(250, 250);

	//Set the window position
	glutInitWindowPosition(100, 100);

	//Create the window
	glutCreateWindow("Dragon");


	//Initialise GLUT
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glShadeModel(GL_FLAT);

	glutDisplayFunc(display);

	//Enter the GLUT event loop
	glutMainLoop();
}




#endif // 0







