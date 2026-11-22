#include "loader.hpp"

#include <nlohmann/json.hpp>
#include <fstream>
#include <unordered_map>
#include <iostream>
#include <vector> 

using namespace std;
using json = nlohmann::json;

// Interface
model_loader::model_loader(string st_mapping) {
    safetensors_mapping = st_mapping;
    weight_file_map();
}

param_offset model_loader::get_param_offset(string param_name) {
    string f_name = get_file_name(param_name);
    f_name = "qwen3-1.7B/" + f_name;
    ifstream f(f_name);
    uint64_t header_bytes;
    f.read(reinterpret_cast<char*>(&header_bytes), sizeof(header_bytes));
    
    f.seekg(8);
    string header_string;
    header_string.resize(header_bytes);
    f.read(header_string.data(), header_bytes);
    json header = json::parse(header_string);

    auto it = header.begin();
    while (it != header.end()) {
        if (it.key() == param_name) {
            long long start = it.value()["data_offsets"][0];
            long long end = it.value()["data_offsets"][1];
            return (param_offset(start + header_bytes + 8, end + header_bytes + 8));
        }
        ++it;
    }

    return param_offset(0, 0);
}

string model_loader::get_file_name(string param_name) {
    if (params_to_file.find(param_name) == params_to_file.end()) {
        return ("");
    }

    return (params_to_file[param_name]);
}

// Helpers
void model_loader::weight_file_map() {
    ifstream f("qwen3-1.7B/model.safetensors.index.json");
    json data = json::parse(f);

    auto it = data["weight_map"].begin();
    while (it != data["weight_map"].end()) {
        params_to_file[it.key()] = it.value();
        ++it;
    }
}




