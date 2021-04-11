#include<Windows.h>    
// first include Windows.h header file which is required    
#include<stdio.h>    
#include<gl/GL.h>   // GL.h header file    
#include<gl/GLU.h> // GLU.h header file    
#include<gl/glut.h>  // glut.h header file from freeglut\include\GL folder    
#include<conio.h>    
#include<stdio.h>    
#include<math.h>    
#include<string.h>    
// Init_OpenGL() function    
void Init_OpenGL()
{
	// set background color to Black    
	glClearColor(0.0, 0.0, 0.0, 0.0);
	// set shade model to Flat    
	glShadeModel(GL_FLAT);
}

// Display_Objects() function    
void Display_Objects(void)
{
	// clearing the window or remove all drawn objects    
	glClear(GL_COLOR_BUFFER_BIT);
	/*glPushMatrix(), which copies the current matrix and adds
	the copy to the top of the stack, and
	glPopMatrix(), which discards the top matrix on the stack*/
	glPushMatrix();
	//the glTranslatef() routine in the display list alters the position of the next object to be drawn    
	glTranslatef(0.0, 0.0, 0.0);
	// set color to object glColor3f(red,green,blue);    
	glColor3f(1.0, 0.8, 0.0);
	// draw a wire tea pot    
	glutWireTeapot(1.0);

	// draw a wire sphere    
	glTranslatef(-2.5, 0.0, 0.0);
	glColor3f(0.0, 1.0, 0.0);
	glutWireSphere(0.8, 30, 30);

	// draw a wire cone    
	glTranslatef(5.0, 0.0, 0.0);
	glColor3f(0.0, 0.6, 1.0);
	glutWireCone(0.8, 1.5, 20, 20);

	// draw a wire cube    
	glTranslatef(-1.0, 1.4, 0.0);
	glColor3f(1.0, 0.3, 0.0);
	glutWireCube(1.0);

	// draw a wire torus    
	glTranslatef(-3.0, 0.4, 0.0);
	glColor3f(1.0, 0.3, 1.0);
	glutWireTorus(0.2, 0.6, 20, 20);

	// draw a text    
	glTranslatef(-2.5, -4.0, 0.0);

	char str[] = { "OpenGL Demo in Visual C++" };

	glColor3f(1.0, 1.0, 1.0);
	// set position to text    
	glRasterPos2f(2.0, 0.0);

	for (int i = 0; i < strlen(str); i++)
	{
		// draw each character    
		glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, str[i]);
	}

	//you can draw many objects here like polygons,lines,triangles etc    

	glPopMatrix();
	glutSwapBuffers();
}
// Reshape() function    
void Reshape(int w, int h)
{
	//adjusts the pixel rectangle for drawing to be the entire new window    
	glViewport(0, 0, (GLsizei)w, (GLsizei)h);
	//matrix specifies the projection transformation    
	glMatrixMode(GL_PROJECTION);
	// load the identity of matrix by clearing it.    
	glLoadIdentity();
	gluPerspective(60.0, (GLfloat)w / (GLfloat)h, 1.0, 20.0);
	//matrix specifies the modelview transformation    
	glMatrixMode(GL_MODELVIEW);
	// again  load the identity of matrix    
	glLoadIdentity();
	// gluLookAt() this function is used to specify the eye.    
	// it is used to specify the coordinates to view objects from a specific position    
	gluLookAt(-0.3, 0.5, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
}

// main function    
int main(int argc, char** argv)
{
	// initialize glut    
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	// set window size    
	glutInitWindowSize(700, 500);
	// set window location    
	glutInitWindowPosition(250, 50);
	// create window with window text    
	glutCreateWindow("OpenGL Demo");
	// call Init_OpenGL() function    
	Init_OpenGL();
	// call glutDisplayFunc() function & pass parameter as Display_Objects() function    
	glutDisplayFunc(Display_Objects);
	// call glutReshapeFunc() function & pass parameter as Reshape() function    
	glutReshapeFunc(Reshape);
	//glutMainLoop() is used to redisplay the objects    
	glutMainLoop();
	return 0;
}