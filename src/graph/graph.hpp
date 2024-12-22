#pragma once
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <optional>

#include "node.hpp"
#include "edge.hpp"

#if 0
template <typename VertexType, typename WeightType>
class Graph {
private:
    std::unordered_map<VertexType, std::unordered_map<VertexType, WeightType>> adjacencyList;

public:
    void addVertex(const VertexType& vertex);
    void addEdge(const VertexType& from, const VertexType& to, const WeightType& weight = WeightType());
    // Other graph operations...
};

#endif

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <vector>
#include <stdexcept>

template <typename VertexProperty = boost::no_property,
          typename EdgeProperty = boost::no_property>
class Graph
{
private:
  using BoostGraph = boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS,
                                           VertexProperty, EdgeProperty>;
  BoostGraph g;
public:
  using vertex_descriptor = typename boost::graph_traits<BoostGraph>::vertex_descriptor;
  using edge_descriptor = typename boost::graph_traits<BoostGraph>::edge_descriptor;

public:
  Graph() = default;

  vertex_descriptor addVertex(const VertexProperty &vp = VertexProperty())
  {
    return boost::add_vertex(vp, g);
  }

  std::pair<edge_descriptor, bool> addEdge(vertex_descriptor source, vertex_descriptor target,
                                           const EdgeProperty &ep = EdgeProperty())
  {
    return boost::add_edge(source, target, ep, g);
  }

  void removeVertex(vertex_descriptor v)
  {
    boost::remove_vertex(v, g);
  }

  void removeEdge(vertex_descriptor source, vertex_descriptor target)
  {
    boost::remove_edge(source, target, g);
  }

  std::size_t vertexCount() const
  {
    return boost::num_vertices(g);
  }

  std::size_t edgeCount() const
  {
    return boost::num_edges(g);
  }

  std::vector<vertex_descriptor> getVertices() const
  {
    std::vector<vertex_descriptor> vertices;
    auto vpair = boost::vertices(g);
    for (auto vit = vpair.first; vit != vpair.second; ++vit)
    {
      vertices.push_back(*vit);
    }
    return vertices;
  }

  std::vector<edge_descriptor> getEdges() const
  {
    std::vector<edge_descriptor> edges;
    auto epair = boost::edges(g);
    for (auto eit = epair.first; eit != epair.second; ++eit)
    {
      edges.push_back(*eit);
    }
    return edges;
  }

  VertexProperty &getVertexProperty(vertex_descriptor v)
  {
    return g[v];
  }

  EdgeProperty &getEdgeProperty(edge_descriptor e)
  {
    return g[e];
  }

  bool hasEdge(vertex_descriptor source, vertex_descriptor target) const
  {
    return boost::edge(source, target, g).second;
  }

  std::vector<vertex_descriptor> getAdjacentVertices(vertex_descriptor v) const
  {
    std::vector<vertex_descriptor> adjacent;
    auto adjPair = boost::adjacent_vertices(v, g);
    for (auto it = adjPair.first; it != adjPair.second; ++it)
    {
      adjacent.push_back(*it);
    }
    return adjacent;
  }

  const BoostGraph &getBoostGraph() const
  {
    return g;
  }

  BoostGraph &getBoostGraph()
  {
    return g;
  }
};