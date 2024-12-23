// clang-format off
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "../graph/graph.hpp"
#include <boost/graph/fruchterman_reingold.hpp>
#include <boost/graph/random_layout.hpp>
#include <random>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
//// clang-format on

template <typename VertexProperty, typename EdgeProperty>
class GraphRenderer {
private:
    Graph<VertexProperty, EdgeProperty> *graph;
    float windowWidth;
    float windowHeight;
    glm::mat4 projection;
    glm::mat4 view;
    float cameraAngle = 0.0f;
    glm::vec3 cameraPosition;
    glm::vec3 cameraTarget;
    float cameraDistance = 10.0f;

public:
    GraphRenderer(Graph<VertexProperty, EdgeProperty> *g, float width, float height)
        : graph(g), windowWidth(width), windowHeight(height) {
        projection = glm::perspective(glm::radians(45.0f), width / height, 0.1f, 100.0f);
        updateCameraPosition();
    }

    void render() {
        glViewport(0, 0, windowWidth, windowHeight);
        glEnable(GL_DEPTH_TEST);

        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        updateCameraPosition();

        view = glm::lookAt(cameraPosition, cameraTarget, glm::vec3(0.0f, 1.0f, 0.0f));

        drawVertices();
        drawEdges();

        // Rotate camera
        cameraAngle += 0.01f;
        if (cameraAngle >= 360.0f) {
            cameraAngle -= 360.0f;
        }
    }

private:
    void updateCameraPosition() {
        glm::vec3 graphCenter = calculateGraphCenter();
        cameraTarget = graphCenter;
        float camX = graphCenter.x + sin(cameraAngle) * cameraDistance;
        float camY = graphCenter.y + cameraDistance * 0.5f;
        float camZ = graphCenter.z + cos(cameraAngle) * cameraDistance;
        cameraPosition = glm::vec3(camX, camY, camZ);
    }

    glm::vec3 calculateGraphCenter() {
        glm::vec3 sum(0.0f);
        const auto& positions = graph->get3DVertexPositions();
        if (positions.empty()) return sum;

        for (const auto& pos : positions) {
            sum.x += std::get<0>(pos);
            sum.y += std::get<1>(pos);
            sum.z += std::get<2>(pos);
        }
        return sum / static_cast<float>(positions.size());
    }

    void drawVertices() {
        glMatrixMode(GL_PROJECTION);
        glLoadMatrixf(glm::value_ptr(projection));
        glMatrixMode(GL_MODELVIEW);
        glLoadMatrixf(glm::value_ptr(view));

        glColor3f(1.0f, 0.0f, 0.0f);
        glPointSize(5.0f);

        glBegin(GL_POINTS);
        for (const auto &pos : graph->get3DVertexPositions()) {
            glVertex3f(std::get<0>(pos), std::get<1>(pos), std::get<2>(pos));
        }
        glEnd();
    }

    void drawEdges() {
        glMatrixMode(GL_PROJECTION);
        glLoadMatrixf(glm::value_ptr(projection));
        glMatrixMode(GL_MODELVIEW);
        glLoadMatrixf(glm::value_ptr(view));

        glLineWidth(1);
        glColor3f(0.4f, 0.4f, 1.0f);

        glBegin(GL_LINES);
        for (const auto &edge : graph->get3DEdgePositions()) {
            auto &source = edge.first;
            auto &target = edge.second;
            glVertex3f(std::get<0>(source), std::get<1>(source), std::get<2>(source));
            glVertex3f(std::get<0>(target), std::get<1>(target), std::get<2>(target));
        }
        glEnd();
    }
};
