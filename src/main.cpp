#include <GL/glut.h>
#include "graph/graph.hpp"
#include "render/graphRenderer.hpp"
#include <iostream>

GLint Width = 800, Height = 600;
GraphRenderer<std::string, int>* renderer;

void Display() {
    renderer->render();
    glutSwapBuffers(); // Swap buffers after rendering
}

void Reshape(GLint w, GLint h) {
    Width = w;
    Height = h;

    glViewport(0, 0, Width, Height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, Width, 0, Height);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void Keyboard(unsigned char key, int x, int y) {
#define ESCAPE '\033'
    if (key == ESCAPE)
        exit(0);
}

void Mouse(int x, int y) {
    // Handle mouse movement here
    glutPostRedisplay(); // Request a redraw of the window
}

int main(int argc, char* argv[]) {
    // Example usage
    Graph<std::string, int> g;

    auto v1 = g.addVertex("Vertex 1");
    auto v2 = g.addVertex("Vertex 2");
    
    std::cout << "Vertex count: " << g.vertexCount() << std::endl;
    
    g.addEdge(v1, v2);
    
    std::cout << "Edge count: " << g.edgeCount() << std::endl;

    glutInit(&argc, argv);
    
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    
    glutInitWindowPosition(0, 0);
    
    glutInitWindowSize(Width, Height);
    
    glutCreateWindow("Graph Visualization");

    renderer = new GraphRenderer<std::string, int>(g, Width, Height);

    glutReshapeFunc(Reshape);
    
    glutDisplayFunc(Display);
    
    glutKeyboardFunc(Keyboard);
    
    glutPassiveMotionFunc(Mouse);

    glutMainLoop();

    delete renderer;

    return 0;
}
