#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../src/engine.hpp"

PYBIND11_MODULE(llm_inference_engine, m) {
    pybind11::class_<Engine>(m, "Engine") 
        .def(pybind11::init<std::vector<int>>())
        .def("forward_pass", &Engine::forward_pass);
}