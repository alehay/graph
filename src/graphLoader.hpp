#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

void read_table(const std::string &filename, std::vector<int> &vertices,
                std::map<int, std::vector<int>> &adj_list) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
  }

  bool is_first_row{true};
  int vertex = 0;
  std::string line;

  while (std::getline(file, line)) {

    std::istringstream iss(line);
    std::cout << "line:{" << line << "}" << std::endl;
    char value;

    int id = 0;

    while (iss >> value) {
      if (is_first_row) {
        if (value == ' ' or value == '[' or value == ']')
          continue;
        std::cout << "[" << value << "]";
        vertices.push_back(static_cast<int>(value - '0'));
      } else {

        if (value == '[') {
          iss >> value;
          vertex = static_cast<int>(value - '0');
          iss >> value;
        } else {
          adj_list[vertex][vertices[id++]] = static_cast<int>(value - '0');
        }
      }
    }

    if (is_first_row) {
      for (auto v : vertices) {
        adj_list[v] = std::vector<int>(vertices.size(), 0);
      }
    }

    is_first_row = false;
  }

  file.close();
}
