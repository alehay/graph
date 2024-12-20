#pragma once

class Node {
public:
    Node(int id, float x, float y);
    int getId() const;
    float getX() const;
    float getY() const;
    void setX(float x);
    void setY(float y);
    // Algorithm-specific data
private:
    int id;
    float x, y;
    // Additional properties
};