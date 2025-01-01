#include "graph/graph.hpp"
#include "render/graphRenderer.hpp"
#include <GL/glew.h>

#include "graphLoader.hpp"
#include <GLFW/glfw3.h>
#include <iostream>

GLint Width = 1200, Height = 800;
GraphRenderer<std::string, boost::property<boost::edge_weight_t, int>>
    *renderer;
Graph<std::string, boost::property<boost::edge_weight_t, int>> *g;

std::random_device rd;                        // Seed source
std::mt19937 gen(rd());                       // Mersenne Twister engine
std::uniform_int_distribution<> dis(10, 100); //

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
    for (int i = 0; i < 2; i++) {
      auto v = g->addVertex(std::to_string(tic++));
      auto rnd_1 = g->getRandomVertex();
      auto rnd_2 = g->getRandomVertex();

      // Check to avoid adding a self-loop for rnd_1
      if (v != rnd_1) {
        g->addEdge(v, rnd_1, dis(gen));
      }

      if (g->getVertices().size() > 20) {
        int random_weight = dis(gen); // Generate random weight

        // Check to avoid adding a self-loop for rnd_2
        if (v != rnd_2) {
          g->addEdge(v, rnd_2, random_weight);
        }
      }
    }
    g->calculateLayout();
  }
}

int main(int argc, char *argv[]) {

  std::string file_path = "./crosstable.txt";
  std::vector<int> vertices;
  std::map<int, std::vector<int>> adj_list;

  read_table(file_path, vertices, adj_list);

  std::cout << "table test " << std::endl;

  for (int i = 0; i < vertices.size(); i++) {
    std::cout << vertices[i] << " ";
    for (int j = 0; j < adj_list[vertices[i]].size(); j++) {
      std::cout << adj_list[vertices[i]][j] << " ";
    }
    std::cout << std::endl;
  }

  glutInit(&argc, argv);

  if (!glfwInit()) {
    std::cerr << "Failed to initialize GLFW" << std::endl;
    return -1;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);

  const GLubyte *renderer_str = glGetString(GL_RENDERER); // GPU or software?
  const GLubyte *vendor_str = glGetString(GL_VENDOR); // e.g. "NVIDIA", "Intel"
  const GLubyte *version_str = glGetString(GL_VERSION);

  //  std::cout << "Renderer: " << renderer_str << std::endl;
  std::cout << "Vendor:   " << vendor_str << std::endl;
  std::cout << "Version:  " << version_str << std::endl;

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

  g = new Graph<std::string>;
  g->populateFromAdjacencyList(vertices, adj_list);
  g->setCube3DLayoutManager(7.0, 5.0, 5.0,
                            5.0); // Adjust these values as needed

  // Add initial vertices and edges

  g->calculateLayout();

  renderer = new GraphRenderer<std::string,
                               boost::property<boost::edge_weight_t, int>>(
      g, Width, Height);

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
