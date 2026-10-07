cmake_minimum_required(VERSION 3.18)

project(s_step_sgd LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(s_step_sgd
    s_step_sgd.cpp
    libsvm_loader.cpp
    prefetch.cpp
    s_step_sgd.cu
    feature_sampling.cu
)

target_include_directories(s_step_sgd PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

find_package(CUDAToolkit REQUIRED)

target_link_libraries(s_step_sgd PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cusolver
)
add_executable(test_factored_gram tests/test_factored_gram.cu s_step_sgd.cu)
target_include_directories(test_factored_gram PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_factored_gram PRIVATE CUDA::cudart CUDA::cublas)

add_executable(test_feature_sampling tests/test_feature_sampling.cpp feature_sampling.cu)
target_include_directories(test_feature_sampling PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_feature_sampling PRIVATE CUDA::cudart CUDA::cublas CUDA::cusolver)
