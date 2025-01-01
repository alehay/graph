// clang-format off
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GL/gl.h>
#include <GL/glu.h>        // For gluProject
#include <GL/glut.h>       // For bitmap font rendering (if desired)
#include <vector>
#include <tuple>
#include <string>
#include <iostream>
#include "../graph/graph.hpp"
#include <boost/graph/fruchterman_reingold.hpp>
#include <boost/graph/random_layout.hpp>
#include <random>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
// clang-format on

#include "../graph/graph.hpp"

class ICamera {
public:
  virtual ~ICamera() = default;
  virtual glm::mat4 getViewMatrix() const = 0;
  virtual void setGraphCenter(const glm::vec3 &center) = 0;
  virtual void update() = 0;
  virtual glm::vec3 getCameraPos() const = 0;
};

class RotatingCamera : public ICamera {
private:
  float cameraAngle = 0.0f;
  float cameraDistance;
  glm::vec3 graphCenter{0.0f};
  glm::vec3 cameraPosition{0.0f};
  glm::vec3 cameraTarget{0.0f};

public:
  explicit RotatingCamera(float distance = 10.0f) : cameraDistance(distance) {}

  void setGraphCenter(const glm::vec3 &center) override {
    graphCenter = center;
  }

  void update() override {
    cameraAngle += 0.01f; // rotate a little each frame
    if (cameraAngle >= 360.0f) {
      cameraAngle -= 360.0f;
    }

    float camX = graphCenter.x + sin(cameraAngle) * cameraDistance;
    float camY = graphCenter.y + cameraDistance * 0.5f;
    float camZ = graphCenter.z + cos(cameraAngle) * cameraDistance;
    cameraPosition = glm::vec3(camX, camY, camZ);
    cameraTarget = graphCenter;
  }

  glm::mat4 getViewMatrix() const override {
    return glm::lookAt(cameraPosition, cameraTarget,
                       glm::vec3(0.0f, 1.0f, 0.0f));
  }

  glm::vec3 getCameraPos() const override { return cameraPosition; }
};

struct EdgeLabelInfo {
  std::tuple<double, double, double> midpoint;
  std::string label;
  double distanceToCamera; // or any additional info you might need
};

class IOpenGLDrawer {
public:
  virtual ~IOpenGLDrawer() = default;

  virtual void drawVertices(
      const std::vector<std::tuple<double, double, double>> &positions) = 0;

  virtual void
  drawEdges(const std::vector<std::pair<std::tuple<double, double, double>,
                                        std::tuple<double, double, double>>>
                &edges) = 0;
  virtual void drawEdgeMidpoints2D(
      const std::vector<std::pair<std::tuple<double, double, double>,
                                  std::tuple<double, double, double>>> &edges,
      const std::vector<std::string> &labels, const glm::vec3 &cameraPos) = 0;
};

class BasicOpenGLDrawer : public IOpenGLDrawer {
public:
  void drawVertices(const std::vector<std::tuple<double, double, double>>
                        &positions) override {
    // Enable point smoothing and blending to get a nicer circle-like point
    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Increase point size
    glPointSize(12.0f);

    // Set vertex color to a more vivid red
    glColor3f(0.8f, 0.2f, 0.2f);

    glBegin(GL_POINTS);
    for (const auto &pos : positions) {
      glVertex3f(static_cast<float>(std::get<0>(pos)),
                 static_cast<float>(std::get<1>(pos)),
                 static_cast<float>(std::get<2>(pos)));
    }
    glEnd();

    // Disable blending / smoothing afterwards if you don’t need them further
    glDisable(GL_BLEND);
    glDisable(GL_POINT_SMOOTH);
  }

  void drawEdges(
      const std::vector<std::pair<std::tuple<double, double, double>,
                                  std::tuple<double, double, double>>> &edges)
      override {
    // Enable line smoothing and blending to get smoother edges
    glEnable(GL_LINE_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Slightly thicker lines
    glLineWidth(2.0f);

    // Set edge color to a calmer bluish tone
    glColor3f(0.3f, 0.3f, 1.0f);

    glBegin(GL_LINES);
    for (const auto &edge : edges) {
      const auto &src = edge.first;
      const auto &tgt = edge.second;

      glVertex3f(static_cast<float>(std::get<0>(src)),
                 static_cast<float>(std::get<1>(src)),
                 static_cast<float>(std::get<2>(src)));
      glVertex3f(static_cast<float>(std::get<0>(tgt)),
                 static_cast<float>(std::get<1>(tgt)),
                 static_cast<float>(std::get<2>(tgt)));
    }
    glEnd();

    // Clean up
    glDisable(GL_BLEND);
    glDisable(GL_LINE_SMOOTH);
  }

  struct ProjectedMidpoint {
    GLdouble winX;
    GLdouble winY;
    GLdouble distance;
  };

  void drawEdgeMidpoints2D(
      const std::vector<std::pair<std::tuple<double, double, double>,
                                  std::tuple<double, double, double>>> &edges,
      const std::vector<std::string> &labels,
      const glm::vec3 &cameraPos) override {
    // 1. Grab the current matrices and viewport info
    GLdouble modelview[16], projection[16];
    GLint viewport[4];

    glGetDoublev(GL_MODELVIEW_MATRIX, modelview);
    glGetDoublev(GL_PROJECTION_MATRIX, projection);
    glGetIntegerv(GL_VIEWPORT, viewport);

    // 2. Compute midpoints in 3D, then project them to 2D
    std::vector<ProjectedMidpoint> projectedPoints;
    projectedPoints.reserve(edges.size());

    for (const auto &edge : edges) {
      const auto &src = edge.first;  // (sx, sy, sz)
      const auto &tgt = edge.second; // (tx, ty, tz)

      // Compute midpoint
      double mx = (std::get<0>(src) + std::get<0>(tgt)) / 2.0;
      double my = (std::get<1>(src) + std::get<1>(tgt)) / 2.0;
      double mz = (std::get<2>(src) + std::get<2>(tgt)) / 2.0;

      double dx = mx - cameraPos.x;
      double dy = my - cameraPos.y;
      double dz = mz - cameraPos.z;
      double distance = std::sqrt(dx * dx + dy * dy + dz * dz);

      // Project to 2D screen coordinates
      GLdouble winX, winY, winZ;
      if (gluProject(mx, my, mz, modelview, projection, viewport, &winX, &winY,
                     &winZ) == GL_TRUE) {
        // If your coordinate system or text rendering expects top-left origin,
        // you might need: winY = viewport[3] - winY;
        projectedPoints.emplace_back(winX, winY, distance);
      }
    }

    // 3. Switch to a 2D orthographic projection
    glMatrixMode(GL_PROJECTION);
    glPushMatrix(); // Save the current projection
    glLoadIdentity();
    glOrtho(0, viewport[2], 0, viewport[3], -1, 1);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix(); // Save the current modelview
    glLoadIdentity();

    // Optionally disable depth test so labels appear on top
    glDisable(GL_DEPTH_TEST);

    // Draw small dots and labels in screen space
    glColor3f(0.0f, 0.0f, 0.5f);
    glPointSize(5.0f);

    glBegin(GL_POINTS);
    for (const auto &[x, y, distance] : projectedPoints) {
      glVertex2d(x, y);
    }
    glEnd();

    // for (const auto &pt : projectedPoints) {

    for (int i = 0; i < labels.size(); ++i) {
      auto &pt = projectedPoints[i];
      float offsetX = 5.0f;
      float offsetY = 5.0f;

      float scale = 100.0f / ((pt.distance + 1.0f)) / 10.0f;

      std::string scale_str = labels[i];

      void *fontToUse = GLUT_BITMAP_HELVETICA_10;

      if (scale >= 0.85f) {
        fontToUse = GLUT_BITMAP_HELVETICA_18;
      } else if (scale >= 0.75f) {
        fontToUse = GLUT_BITMAP_HELVETICA_12;
      }

      glPushMatrix();
      {
        // Translate to (winX + offsetX, winY + offsetY)
        glTranslatef(pt.winX + offsetX, pt.winY + offsetY, 0.0f);

        // Scale in 2D
        glScalef(scale, scale, 1.0f);

        // Now set the raster pos at the local origin
        glRasterPos2f(0.0f, 0.0f);

        // Draw text at (0,0) in scaled space
        for (char c : scale_str) {
          glutBitmapCharacter(fontToUse, c);
        }
      }
      glPopMatrix();
    }

    // Restore state
    glEnable(GL_DEPTH_TEST);

    glPopMatrix(); // Restore the old modelview
    glMatrixMode(GL_PROJECTION);
    glPopMatrix(); // Restore the old projection
    glMatrixMode(GL_MODELVIEW);
  }
};

template <typename VertexProperty, typename EdgeProperty> class GraphRenderer {
private:
  Graph<VertexProperty, EdgeProperty> *graph;
  float windowWidth;
  float windowHeight;
  glm::mat4 projection;

  // Composition via interfaces
  std::unique_ptr<ICamera> camera;
  std::unique_ptr<IOpenGLDrawer> drawer;

public:
  GraphRenderer(Graph<VertexProperty, EdgeProperty> *g, float width,
                float height, std::unique_ptr<ICamera> cam,
                std::unique_ptr<IOpenGLDrawer> draw)
      : graph(g), windowWidth(width), windowHeight(height),
        camera(std::move(cam)), drawer(std::move(draw)) {
    projection = glm::perspective(glm::radians(45.0f),
                                  windowWidth / windowHeight, 0.1f, 100.0f);
  }

  GraphRenderer(Graph<VertexProperty, EdgeProperty> *g, float width,
                float height)
      : graph(g), windowWidth(width), windowHeight(height) {

    camera = std::make_unique<RotatingCamera>(10.0f);
    drawer = std::make_unique<BasicOpenGLDrawer>();
  }

  void render() {
    // Setup viewport, depth test, clear color
    glViewport(0, 0, static_cast<GLsizei>(windowWidth),
               static_cast<GLsizei>(windowHeight));
    glEnable(GL_DEPTH_TEST);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Update camera
    camera->setGraphCenter(calculateGraphCenter());
    camera->update();
    projection = glm::perspective(glm::radians(45.0f),
                                  static_cast<float>(windowWidth) /
                                      static_cast<float>(windowHeight),
                                  0.1f, 100.0f);

    // Setup projection
    glMatrixMode(GL_PROJECTION);
    glLoadMatrixf(glm::value_ptr(projection));

    // Setup model/view
    glMatrixMode(GL_MODELVIEW);
    glm::mat4 view = camera->getViewMatrix();
    glLoadMatrixf(glm::value_ptr(view));

    // Delegate drawing
    drawer->drawVertices(graph->get3DVertexPositions());

    std::vector<std::pair<std::tuple<double, double, double>,
                          std::tuple<double, double, double>>>
        vec;
    std::vector<std::string> lables;

    auto edgeInfoVec = graph->get3DEdgePositions();

    for (const auto &edgeInfo : edgeInfoVec) {
      // Method 1: Using boost::geometry::get
      double src_x = boost::geometry::get<0>(edgeInfo.sourcePos);
      double src_y = boost::geometry::get<1>(edgeInfo.sourcePos);
      double src_z = boost::geometry::get<2>(edgeInfo.sourcePos);

      double tgt_x = boost::geometry::get<0>(edgeInfo.targetPos);
      double tgt_y = boost::geometry::get<1>(edgeInfo.targetPos);
      double tgt_z = boost::geometry::get<2>(edgeInfo.targetPos);

      // Create tuples for source and target positions
      std::tuple<double, double, double> sourceTuple(src_x, src_y, src_z);
      std::tuple<double, double, double> targetTuple(tgt_x, tgt_y, tgt_z);

      // Add the pair to the vector
      vec.emplace_back(std::make_pair(sourceTuple, targetTuple));

      lables.emplace_back(std::to_string(edgeInfo.weight));


    }
    drawer->drawEdges(vec);
    drawer->drawEdgeMidpoints2D(vec, lables, camera->getCameraPos());
  }

private:
  // Utility to find center of graph for the camera
  glm::vec3 calculateGraphCenter() {
    glm::vec3 sum(0.0f);
    const auto &positions = graph->get3DVertexPositions();
    if (positions.empty()) {
      return sum;
    }
    for (const auto &pos : positions) {
      sum.x += static_cast<float>(std::get<0>(pos));
      sum.y += static_cast<float>(std::get<1>(pos));
      sum.z += static_cast<float>(std::get<2>(pos));
    }
    return sum / static_cast<float>(positions.size());
  }
};
