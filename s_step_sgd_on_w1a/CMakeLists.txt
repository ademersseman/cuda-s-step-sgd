cmake_minimum_required(VERSION 3.18)

project(s_step_sgd LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(s_step_sgd
    s_step_sgd.cpp
    libsvm_loader.cpp
    prefetch.cpp
    s_step_sgd.cu
)

target_include_directories(s_step_sgd PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

find_package(CUDAToolkit REQUIRED)

target_link_libraries(s_step_sgd PRIVATE
    CUDA::cudart
    CUDA::cublas
)