#include "graph/graph.hpp"
#include "render/graphRenderer.hpp"
#include <GLFW/glfw3.h>
#include <iostream>

GLint Width = 1200, Height = 800;
GraphRenderer<std::string, int> *renderer;
Graph<std::string, int> *g;

void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
  Width = width;
  Height = height;
  glViewport(0, 0, width, height);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, width, height, 0, -1, 1);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glMatrixMode(GL_MODELVIEW);
}

int tic = 0;

void key_callback(GLFWwindow *window, int key, int scancode, int action,
                  int mods) {
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    glfwSetWindowShouldClose(window, GLFW_TRUE);
  if (key == GLFW_KEY_Q && action == GLFW_PRESS) {
    auto v = g->addVertex(std::to_string(tic));
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

  GLFWwindow *window =
      glfwCreateWindow(Width, Height, "Graph Visualization", NULL, NULL);
  if (!window) {
    std::cerr << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return -1;
  }

  glfwGetFramebufferSize(window, &Width, &Height);
  framebuffer_size_callback(window, Width, Height);

  glfwMakeContextCurrent(window);
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
  glfwSetKeyCallback(window, key_callback);

  g = new Graph<std::string, int>;
  auto layoutManager =
      std::make_unique<FruchtermanReingoldLayout<std::string, int>>(Width,
                                                                    Height);
  g->setLayoutManager(std::move(layoutManager));

  // Add vertices and edges to the graph...
  // (Your existing graph setup code here)

  renderer = new GraphRenderer<std::string, int>(g, Width, Height);

  while (!glfwWindowShouldClose(window)) {
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
