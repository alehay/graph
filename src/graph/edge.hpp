#pragma once

#include "node.hpp"

class Edge {
public:
    Edge(Node* src, Node* dest, float weight = 1.0f);
    Node* getSource() const;
    Node* getDestination() const;
    float getWeight() const;
    void setWeight(float weight);
    // Additional properties
private:
    Node* source;
    Node* destination;
    float weight;
};