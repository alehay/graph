#include "../graph/graph.hpp"
#include <GL/glut.h>
#include <boost/graph/fruchterman_reingold.hpp>
#include <boost/graph/random_layout.hpp>
#include <random>

template <typename VertexProperty, typename EdgeProperty>
class GraphRenderer {
private:
    Graph<VertexProperty, EdgeProperty>& graph;
    float windowWidth;
    float windowHeight;
    std::vector<boost::rectangle_topology<>::point_type> positions;

public:
    GraphRenderer(Graph<VertexProperty, EdgeProperty>& g, float width, float height)
        : graph(g), windowWidth(width), windowHeight(height) {
        calculateLayout();
    }

    void render() {
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        GLenum error = glGetError();
        if (error != GL_NO_ERROR) {
            printf("OpenGL error: %d\n", error);
        }

        drawVertices();
        drawEdges();
        glFlush();
    }

private:
    void calculateLayout() {
        auto& g = graph.getBoostGraph();
        positions.resize(boost::num_vertices(g));

        // Generate initial random layout
        boost::minstd_rand gen;
        boost::rectangle_topology<> topology(gen, 10, 10, windowWidth - 10, windowHeight - 10);
        boost::random_graph_layout(
            g,
            make_iterator_property_map(positions.begin(), get(boost::vertex_index, g)),
            topology);

        // Apply Fruchterman-Reingold force-directed algorithm
        boost::fruchterman_reingold_force_directed_layout(
            g,
            make_iterator_property_map(positions.begin(), get(boost::vertex_index, g)),
            topology,
            cooling(boost::linear_cooling<double>(100)));
    }

void drawVertices() {
    glColor3f(1.0f, 0.0f, 0.0f);
    glPointSize(15.0f);
    
    glBegin(GL_POINTS);
    for (const auto& vertex : graph.getVertices()) {
        auto& pos = positions[vertex];
        glVertex2f(pos[0], pos[1]);
    }
    glEnd();
}

void drawEdges() {
    glLineWidth(1);
    glColor3f(0.4f, 0.4f, 1.0f);
    
    glBegin(GL_LINES);
    for (const auto& edge : graph.getEdges()) {
        auto source = boost::source(edge, graph.getBoostGraph());
        auto target = boost::target(edge, graph.getBoostGraph());
        auto& sourcePos = positions[source];
        auto& targetPos = positions[target];
        
        glVertex2f(sourcePos[0], sourcePos[1]);
        glVertex2f(targetPos[0], targetPos[1]);
    }
    glEnd();
}

};