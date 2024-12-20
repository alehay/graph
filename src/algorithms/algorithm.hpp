#pragma once

#include "../graph/graph.hpp"

template <typename VertexType, typename WeightType>
class GraphAlgorithm {
public:
    virtual void execute(const Graph<VertexType, WeightType>& graph) = 0;
    virtual ~GraphAlgorithm() = default;
};


template <typename VertexType, typename WeightType>
class PrimAlgorithm : public GraphAlgorithm<VertexType, WeightType> {
public:
    void execute(const Graph<VertexType, WeightType>& graph) override;
};

// Similar classes for Kruskal, Boruvka, Floyd-Warshall, Dijkstra, Bellman-Ford, Johnson, Levite, and Yen algorithms
