cmake_minimum_required(VERSION 3.18)
project(inference_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(engine_core STATIC
    src/loader.cpp 
    src/engine.cpp 
)

target_include_directories(engine_core PUBLIC third_party Eigen)
set_target_properties(engine_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(forward src/main.cpp)
target_link_libraries(forward PUBLIC engine_core)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(llm_inference_engine bindings/bindings.cpp)
target_link_libraries(llm_inference_engine PUBLIC engine_core)