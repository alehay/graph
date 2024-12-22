#include "../graph/graph.hpp"
#include <GL/glut.h>
#include <boost/graph/fruchterman_reingold.hpp>
#include <boost/graph/random_layout.hpp>
#include <random>

template <typename VertexProperty, typename EdgeProperty> class GraphRenderer {
private:
  Graph<VertexProperty, EdgeProperty> *graph;
  float windowWidth;
  float windowHeight;
  std::vector<boost::rectangle_topology<>::point_type> positions;

public:
  GraphRenderer(Graph<VertexProperty, EdgeProperty> *g, float width,
                float height)
      : graph(g), windowWidth(width), windowHeight(height) {}

  void render() {
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
      printf("OpenGL error: %d\n", error);
    }
    graph->calculateLayout();
    drawVertices();
    drawEdges();
    glFlush();
  }

private:
  void drawVertices() {
    glColor3f(1.0f, 0.0f, 0.0f);
    glPointSize(15.0f);

    glBegin(GL_POINTS);

    // good
    for (const auto &[x, y] : graph->getVertexPositions()) {

      glVertex2f(x, y);
    }
    glEnd();
  }

  void drawEdges() {
    glLineWidth(1);
    glColor3f(0.4f, 0.4f, 1.0f);

    glBegin(GL_LINES);

    for (const auto &edge : graph->get2DEdgePositions()) {
      auto &source = edge.first;
      auto &target = edge.second;

      glVertex2f(source.first, source.second);
      glVertex2f(target.first, target.second);
    }

    glEnd();
  }
};