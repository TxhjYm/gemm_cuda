cmake_minimum_required(VERSION 3.18)
project(gemmgpu VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

file(GLOB SOURCES_CPU CONFIGURE_DEPENDS src/cpu/*.cpp)
file(GLOB SOURCES_GPU CONFIGURE_DEPENDS src/cuda/*.cu)

add_executable(gemmtestgpu
    src/main.cu
    ${SOURCES_CPU}
    ${SOURCES_GPU}
)

target_include_directories(gemmtestgpu
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(gemmtestgpu PRIVATE CUDA::cudart)