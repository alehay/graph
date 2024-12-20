#pragma once

class GraphDrawer {
public:
    GraphDrawer(sf::RenderWindow& window);
    void draw(const Graph& graph);
    void updateTransformations(float rotation, float scale);
    // Additional drawing functions
private:
//    sf::RenderWindow& window;
//    sf::Transform transform;
    // Drawing helpers
};