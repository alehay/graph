#pragma once

#include "graph/graph.hpp"
class FileIO {
public:
    template <typename VertexType, typename WeightType>
    static Graph<VertexType, WeightType> readGraphFromFile(const std::string& filename);

    template <typename VertexType, typename WeightType>
    static void writeGraphToFile(const Graph<VertexType, WeightType>& graph, const std::string& filename);
};
