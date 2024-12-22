#pragma once
#include <chrono>
#include <optional>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "edge.hpp"
#include "node.hpp"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>

#include <boost/graph/fruchterman_reingold.hpp>
#include <boost/graph/random_layout.hpp>
#include <stdexcept>
#include <vector>

template <typename VertexProperty, typename EdgeProperty> struct IGraphTraits {
  using BoostGraph =
      boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS,
                            VertexProperty, EdgeProperty>;

  using vertex_descriptor =
      typename boost::graph_traits<BoostGraph>::vertex_descriptor;
  using edge_descriptor =
      typename boost::graph_traits<BoostGraph>::edge_descriptor;
};

template <typename VertexProperty, typename EdgeProperty>
class IGraphStructure {
  using Traits = IGraphTraits<VertexProperty, EdgeProperty>;
  using vertex_descriptor = typename Traits::vertex_descriptor;
  using edge_descriptor = typename Traits::edge_descriptor;

public:
  virtual ~IGraphStructure() = default;
  virtual std::size_t vertexCount() const = 0;
  virtual std::size_t edgeCount() const = 0;
  virtual std::vector<vertex_descriptor> getVertices() const = 0;
  virtual std::vector<edge_descriptor> getEdges() const = 0;
  virtual bool hasEdge(vertex_descriptor source,
                       vertex_descriptor target) const = 0;
  virtual std::vector<vertex_descriptor>
  getAdjacentVertices(vertex_descriptor v) const = 0;
};

template <typename VertexProperty, typename EdgeProperty>
class BoostGraphStructure
    : public IGraphStructure<VertexProperty, EdgeProperty> {
private:
  using Traits = IGraphTraits<VertexProperty, EdgeProperty>;
  using BoostGraph = typename Traits::BoostGraph;
  using vertex_descriptor = typename Traits::vertex_descriptor;
  using edge_descriptor = typename Traits::edge_descriptor;

  BoostGraph g;

public:
    std::size_t vertexCount() const override { 
        return boost::num_vertices(g); 
    }

    std::size_t edgeCount() const override { 
        return boost::num_edges(g); 
    }

    std::vector<vertex_descriptor> getVertices() const override {
        std::vector<vertex_descriptor> vertices;
        auto vpair = boost::vertices(g);
        for (auto vit = vpair.first; vit != vpair.second; ++vit) {
            vertices.push_back(*vit);
        }
        return vertices;
    }

    std::vector<edge_descriptor> getEdges() const override {
        std::vector<edge_descriptor> edges;
        auto epair = boost::edges(g);
        for (auto eit = epair.first; eit != epair.second; ++eit) {
            edges.push_back(*eit);
        }
        return edges;
    }

    bool hasEdge(vertex_descriptor source, vertex_descriptor target) const override {
        return boost::edge(source, target, g).second;
    }

    std::vector<vertex_descriptor> getAdjacentVertices(vertex_descriptor v) const override {
        std::vector<vertex_descriptor> adjacent;
        auto adjPair = boost::adjacent_vertices(v, g);
        for (auto it = adjPair.first; it != adjPair.second; ++it) {
            adjacent.push_back(*it);
        }
        return adjacent;
    }

    // Additional method to access the underlying Boost graph
    BoostGraph& getBoostGraph() { return g; }
    const BoostGraph& getBoostGraph() const { return g; }

};

template <typename VertexProperty, typename EdgeProperty>
class IVertexOperations {
  using Traits = IGraphTraits<VertexProperty, EdgeProperty>;
  using BoostGraph = typename Traits::BoostGraph;
  using vertex_descriptor = typename Traits::vertex_descriptor;
  using edge_descriptor = typename Traits::edge_descriptor;

public:
  virtual ~IVertexOperations() = default;
  virtual vertex_descriptor addVertex(const VertexProperty &vp) = 0;
  virtual void removeVertex(vertex_descriptor v) = 0;
  virtual VertexProperty &getVertexProperty(vertex_descriptor v) = 0;
};

template <typename VertexProperty, typename EdgeProperty>
class BoostVertexOperations
    : public IVertexOperations<VertexProperty, EdgeProperty> {

private:
    using Traits = IGraphTraits<VertexProperty, EdgeProperty>;
    using BoostGraph = typename Traits::BoostGraph;
    using vertex_descriptor = typename Traits::vertex_descriptor;
    using edge_descriptor = typename Traits::edge_descriptor;

    BoostGraphStructure<VertexProperty, EdgeProperty>& graphStructure;

public:
    BoostVertexOperations(
        BoostGraphStructure<VertexProperty, EdgeProperty>& structure)
        : graphStructure(structure) {}

    vertex_descriptor addVertex(const VertexProperty& vp) override {
        return boost::add_vertex(vp, graphStructure.getBoostGraph());
    }

    void removeVertex(vertex_descriptor v) override {
        boost::remove_vertex(v, graphStructure.getBoostGraph());
    }

    VertexProperty& getVertexProperty(vertex_descriptor v) override {
        return graphStructure.getBoostGraph()[v];
    }
};


template <typename VertexProperty, typename EdgeProperty>
class IEdgeOperations {
  using Traits = IGraphTraits<VertexProperty, EdgeProperty>;
  using vertex_descriptor = typename Traits::vertex_descriptor;
  using edge_descriptor = typename Traits::edge_descriptor;

public:
  virtual ~IEdgeOperations() = default;
  virtual std::pair<edge_descriptor, bool> addEdge(vertex_descriptor source,
                                                   vertex_descriptor target,
                                                   const EdgeProperty &ep) = 0;
  virtual void removeEdge(vertex_descriptor source,
                          vertex_descriptor target) = 0;
  virtual EdgeProperty &getEdgeProperty(edge_descriptor e) = 0;
};

template <typename VertexProperty, typename EdgeProperty>
class BoostEdgeOperations
    : public IEdgeOperations<VertexProperty, EdgeProperty> {
private:
    using Traits = IGraphTraits<VertexProperty, EdgeProperty>;
    using BoostGraph = typename Traits::BoostGraph;
    using vertex_descriptor = typename Traits::vertex_descriptor;
    using edge_descriptor = typename Traits::edge_descriptor;

    BoostGraphStructure<VertexProperty, EdgeProperty>& graphStructure;

public:
    BoostEdgeOperations(
        BoostGraphStructure<VertexProperty, EdgeProperty>& structure)
        : graphStructure(structure) {}

    std::pair<edge_descriptor, bool> addEdge(vertex_descriptor source,
                                             vertex_descriptor target,
                                             const EdgeProperty& ep) override {
        return boost::add_edge(source, target, ep, graphStructure.getBoostGraph());
    }

    void removeEdge(vertex_descriptor source,
                    vertex_descriptor target) override {
        boost::remove_edge(source, target, graphStructure.getBoostGraph());
    }

    EdgeProperty& getEdgeProperty(edge_descriptor e) override {
        return graphStructure.getBoostGraph()[e];
    }
};


template <typename VertexProperty, typename EdgeProperty>
class IRandomSelector {

  using Traits = IGraphTraits<VertexProperty, EdgeProperty>;
  using vertex_descriptor = typename Traits::vertex_descriptor;
  using edge_descriptor = typename Traits::edge_descriptor;

public:
  virtual ~IRandomSelector() = default;
  virtual vertex_descriptor
  getRandomVertex(const std::vector<vertex_descriptor> &vertices) const = 0;
};

template <typename VertexProperty, typename EdgeProperty>
class DefaultRandomSelector
    : public IRandomSelector<VertexProperty, EdgeProperty> {
  using Traits = IGraphTraits<VertexProperty, EdgeProperty>;
  using vertex_descriptor = typename Traits::vertex_descriptor;
  using edge_descriptor = typename Traits::edge_descriptor;

public:
  vertex_descriptor getRandomVertex(
      const std::vector<vertex_descriptor> &vertices) const override {
    if (vertices.empty()) {
      throw std::runtime_error("Graph is empty");
    }

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<std::size_t> distribution(0, vertices.size() -
                                                                   1);

    return vertices[distribution(generator)];
  }
};

template <typename VertexProperty = boost::no_property,
          typename EdgeProperty = boost::no_property>
class Graph {
private:
    std::unique_ptr<IGraphStructure<VertexProperty, EdgeProperty>> structure;
    std::unique_ptr<IVertexOperations<VertexProperty, EdgeProperty>> vertexOps;
    std::unique_ptr<IEdgeOperations<VertexProperty, EdgeProperty>> edgeOps;
    std::unique_ptr<IRandomSelector<VertexProperty, EdgeProperty>> randomSelector;

public:
    using Traits = IGraphTraits<VertexProperty, EdgeProperty>;
    using vertex_descriptor = typename Traits::vertex_descriptor;
    using edge_descriptor = typename Traits::edge_descriptor;
    using BoostGraph = typename Traits::BoostGraph;

public:
    Graph()
        : structure(std::make_unique<BoostGraphStructure<VertexProperty, EdgeProperty>>()),
          vertexOps(std::make_unique<BoostVertexOperations<VertexProperty, EdgeProperty>>(
              *static_cast<BoostGraphStructure<VertexProperty, EdgeProperty>*>(structure.get()))),
          edgeOps(std::make_unique<BoostEdgeOperations<VertexProperty, EdgeProperty>>(
              *static_cast<BoostGraphStructure<VertexProperty, EdgeProperty>*>(structure.get()))),
          randomSelector(std::make_unique<DefaultRandomSelector<VertexProperty, EdgeProperty>>()) {}

    vertex_descriptor addVertex(const VertexProperty& vp = VertexProperty()) {
        return vertexOps->addVertex(vp);
    }

    vertex_descriptor getRandomVertex() const {
        auto vertices = structure->getVertices();
        return randomSelector->getRandomVertex(vertices);
    }

    std::pair<edge_descriptor, bool> addEdge(vertex_descriptor source, vertex_descriptor target,
                                             const EdgeProperty& ep = EdgeProperty()) {
        return edgeOps->addEdge(source, target, ep);
    }

    std::size_t vertexCount() const { return structure->vertexCount(); }

    std::size_t edgeCount() const { return structure->edgeCount(); }

    std::vector<vertex_descriptor> getVertices() const { return structure->getVertices(); }

    std::vector<edge_descriptor> getEdges() const { return structure->getEdges(); }

    const BoostGraph& getBoostGraph() const {
        return static_cast<const BoostGraphStructure<VertexProperty, EdgeProperty>*>(structure.get())->getBoostGraph();
    }
};

#if 0
template <typename VertexProperty = boost::no_property,
          typename EdgeProperty = boost::no_property>
class Graph {
private:
  using BoostGraph =
      boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS,
                            VertexProperty, EdgeProperty>;
  BoostGraph g;

public:
  using vertex_descriptor =
      typename boost::graph_traits<BoostGraph>::vertex_descriptor;
  using edge_descriptor =
      typename boost::graph_traits<BoostGraph>::edge_descriptor;

public:
  Graph() = default;

  vertex_descriptor addVertex(const VertexProperty &vp = VertexProperty()) {
    return boost::add_vertex(vp, g);
  }

  std::pair<edge_descriptor, bool>
  addEdge(vertex_descriptor source, vertex_descriptor target,
          const EdgeProperty &ep = EdgeProperty()) {
    return boost::add_edge(source, target, ep, g);
  }

  void removeVertex(vertex_descriptor v) { boost::remove_vertex(v, g); }

  void removeEdge(vertex_descriptor source, vertex_descriptor target) {
    boost::remove_edge(source, target, g);
  }

  std::size_t vertexCount() const { return boost::num_vertices(g); }

  std::size_t edgeCount() const { return boost::num_edges(g); }

  std::vector<vertex_descriptor> getVertices() const {
    std::vector<vertex_descriptor> vertices;
    auto vpair = boost::vertices(g);
    for (auto vit = vpair.first; vit != vpair.second; ++vit) {
      vertices.push_back(*vit);
    }
    return vertices;
  }

  std::vector<edge_descriptor> getEdges() const {
    std::vector<edge_descriptor> edges;
    auto epair = boost::edges(g);
    for (auto eit = epair.first; eit != epair.second; ++eit) {
      edges.push_back(*eit);
    }
    return edges;
  }

  VertexProperty &getVertexProperty(vertex_descriptor v) { return g[v]; }

  EdgeProperty &getEdgeProperty(edge_descriptor e) { return g[e]; }

  bool hasEdge(vertex_descriptor source, vertex_descriptor target) const {
    return boost::edge(source, target, g).second;
  }

  std::vector<vertex_descriptor>
  getAdjacentVertices(vertex_descriptor v) const {
    std::vector<vertex_descriptor> adjacent;
    auto adjPair = boost::adjacent_vertices(v, g);
    for (auto it = adjPair.first; it != adjPair.second; ++it) {
      adjacent.push_back(*it);
    }
    return adjacent;
  }

  const BoostGraph &getBoostGraph() const { return g; }

  BoostGraph &getBoostGraph() { return g; }

  vertex_descriptor getRandomVertex() const {
    if (vertexCount() == 0) {
      throw std::runtime_error("Graph is empty");
    }

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<std::size_t> distribution(0,
                                                            vertexCount() - 1);

    auto vertices = getVertices();
    return vertices[distribution(generator)];
  }

  std::optional<edge_descriptor>
  getEdgeConnecting(vertex_descriptor source, vertex_descriptor target) const {
    auto edge_pair = boost::edge(source, target, g);
    if (edge_pair.second) {
      return edge_pair.first;
    }
    return std::nullopt;
  }
};

#endif