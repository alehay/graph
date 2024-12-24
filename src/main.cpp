#include "graph/graph.hpp"
#include "render/graphRenderer.hpp"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>

GLint Width = 1200, Height = 800;
GraphRenderer<std::string, int> *renderer;
Graph<std::string, int> *g;

void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
  Width = width;
  Height = height;
  glViewport(0, 0, width, height);

  // Update projection matrix for 3D
//  glMatrixMode(GL_PROJECTION);
//  glLoadIdentity();
//  gluPerspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);

//  glMatrixMode(GL_MODELVIEW);
//  glLoadIdentity();
}

int tic = 0;

void key_callback(GLFWwindow *window, int key, int scancode, int action,
                  int mods) {
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    glfwSetWindowShouldClose(window, GLFW_TRUE);
  if (key == GLFW_KEY_Q && action == GLFW_PRESS) {
    auto v = g->addVertex(std::to_string(tic++));
    auto rnd_1 = g->getRandomVertex();
    auto rnd_2 = g->getRandomVertex();
    g->addEdge(v, rnd_1);
    if (g->getVertices().size() > 10) {
      g->addEdge(v, rnd_2);
    }
    g->calculateLayout();
  }
}

int main() {
  if (!glfwInit()) {
    std::cerr << "Failed to initialize GLFW" << std::endl;
    return -1;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);

  GLFWwindow *window =
      glfwCreateWindow(Width, Height, "3D Graph Visualization", NULL, NULL);
  if (!window) {
    std::cerr << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return -1;
  }

  glfwMakeContextCurrent(window);
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
  glfwSetKeyCallback(window, key_callback);

  if (glewInit() != GLEW_OK) {
    std::cerr << "Failed to initialize GLEW" << std::endl;
    return -1;
  }

  g = new Graph<std::string, int>;
  g->setCube3DLayoutManager(7.0, 5.0, 5.0,
                            5.0); // Adjust these values as needed

  // Add initial vertices and edges
  for (int i = 0; i < 10; ++i) {
    g->addVertex(std::to_string(i));
  }
  for (int i = 0; i < 15; ++i) {
    g->addEdge(g->getRandomVertex(), g->getRandomVertex());
  }
  g->calculateLayout();

  renderer = new GraphRenderer<std::string, int>(g, Width, Height);

  // Enable depth testing
  glEnable(GL_DEPTH_TEST);


  while (!glfwWindowShouldClose(window)) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    renderer->render();

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  delete renderer;
  delete g;

  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}
