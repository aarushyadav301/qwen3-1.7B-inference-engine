#ifndef ENGINE 
#define ENGINE 

#include "loader.hpp"

#include <iostream> 
#include <vector>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

using KVMatrix = Matrix<float, Dynamic, 128, RowMajor>;
using KVTransposed = Matrix<float, 128, Dynamic>;
using WeightedValueMatrix = Matrix<float, Dynamic, 128, RowMajor>;
using ConcatenatedValueMatrix = Matrix<float, Dynamic, 2048, RowMajor>;
using ResidualStream = Matrix<float, Dynamic, 2048, RowMajor>;

using HeapMatrix = Matrix<float, Dynamic, Dynamic>;
using RowVec = RowVector<float, Dynamic>;

class Engine {
public:
    Engine(vector<int> ids);
    vector<int> forward_pass();

private:
    vector<int> input_ids;
};

#endif