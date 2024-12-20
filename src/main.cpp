#include "graph/graph.hpp"

#include <iostream>

int main()
{
  // Example usage
  Graph<std::string, int> g;

  auto v1 = g.addVertex("Vertex 1");
  auto v2 = g.addVertex("Vertex 2");
  auto v11 = g.addVertex("Vertex 1");

  std::cout << "vertext count: " << g.vertexCount() << std::endl;
  std::cout << "edge count: " << g.edgeCount() << std::endl;

  g.addEdge(v1, v2);

  std::cout << "vertext count: " << g.vertexCount() << std::endl;
  std::cout << "edge count: " << g.edgeCount() << std::endl;

  g.addEdge(v1, v11);
  std::cout << "vertext count: " << g.vertexCount() << std::endl;
  std::cout << "edge count: " << g.edgeCount() << std::endl;
}