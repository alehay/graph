#include "graph/graph.hpp"
#include "render/graphRenderer.hpp"
#include <GL/glut.h>
#include <iostream>

GLint Width = 800, Height = 600;
GraphRenderer<std::string, int> *renderer;
Graph<std::string, int> *g;

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
  if (key == 'q') {
    auto v = g->addVertex("");
    auto rnd_1 = g->getRandomVertex();
    auto rnd_2 = g->getRandomVertex();

    g->addEdge(v, rnd_1);
    g->addEdge(v, rnd_2);
  }
}

void timer(int v) {
  glLoadIdentity();
  glutPostRedisplay();
  glutTimerFunc(10000 / 60.0, timer, v);
  std::cout << "tic" << std::endl;
}

void Mouse(int x, int y) {
  // Handle mouse movement here
  glutPostRedisplay(); // Request a redraw of the window
}

int main(int argc, char *argv[]) {
  std::random_device rd;
  std::mt19937 gen(rd());

  // Example usage
  g = new Graph<std::string, int>;

  auto layoutManager =
      std::make_unique<FruchtermanReingoldLayout<std::string, int>>(800, 600);
  g->setLayoutManager(std::move(layoutManager));

  auto v1 = g->addVertex("Vertex 1");
  auto v2 = g->addVertex("Vertex 2");

  std::cout << "Vertex count: " << g->vertexCount() << std::endl;

  g->addEdge(v1, v2);

  auto v3 = g->addVertex("Vertex 3");

  g->addEdge(v1, v3);

  auto v4 = g->addVertex("Vertex 4");
  auto v5 = g->addVertex("Vertex 1");
  auto v6 = g->addVertex("Vertex 1");
  g->addEdge(v4, v5);

  using vertex_descriptor = typename Graph<std::string, int>::vertex_descriptor;

  std::vector<vertex_descriptor> vert_1;
  for (int i = 10; i >= 0; i--) {
    auto v = g->addVertex("Vertex " + std::to_string(i));
    g->addEdge(v, v1);
    vert_1.push_back(v);
  }

  std::vector<vertex_descriptor> vert_2;

  for (int i = 10; i >= 0; i--) {
    auto v = g->addVertex("Vertex " + std::to_string(i));
    g->addEdge(v, v3);
    vert_2.push_back(v);
  }

  std::uniform_int_distribution<> distrib(1, 10);

  for (size_t i = 0; i < 40; i++) {
    g->addEdge(vert_1[distrib(gen)], vert_2[distrib(gen)]);
  }

  std::cout << "Edge count: " << g->edgeCount() << std::endl;

  glutInit(&argc, argv);

  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);

  glutInitWindowPosition(0, 0);

  glutInitWindowSize(Width, Height);

  glutCreateWindow("Graph Visualization");

  renderer = new GraphRenderer<std::string, int>(g, Width, Height);

  glutReshapeFunc(Reshape);

  glutDisplayFunc(Display);

  glutKeyboardFunc(Keyboard);
  glutTimerFunc(1000, timer, 0);

  glutPassiveMotionFunc(Mouse);

  glutMainLoop();

  delete renderer;
  delete g;

  return 0;
}
