#include <iostream>
#include <map>
#include <vector>


int main() {

  std::cout << "Hello World!" << std::endl;

  std::map<int, std::vector<uint32_t>> my_map;

  // Get the vector for the key 10.
  std::vector<uint32_t>& my_vector = my_map[10];

  // Add the values 1, 2, and 3 to the vector using emplace().
  my_vector.emplace_back(1);
  my_vector.emplace_back(2);
  my_vector.emplace_back(3);

  // Print the vector.
  std::cout << "my_vector: ";
  for (auto& i : my_vector) {
    std::cout << i << " ";
  }

  return 0;

}