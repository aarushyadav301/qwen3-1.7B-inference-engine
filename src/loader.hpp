#ifndef LOADER 
#define LOADER 

#include <string>
#include <unordered_map>
#include <nlohmann/json.hpp>

using namespace std;
using json = nlohmann::json;

struct param_offset {
    uint64_t start;
    uint64_t end;

    param_offset(uint64_t s, uint64_t e) {
        start = s;
        end = e;
    }
};

class model_loader {
    public:
        model_loader(string safetensors_mapping = "model.safetensors.index.json");
        param_offset get_param_offset(string param_name);
        string get_file_name(string param_name);

    private:
        void weight_file_map();

        unordered_map<string, string> params_to_file;
        string safetensors_mapping;
};

#endif