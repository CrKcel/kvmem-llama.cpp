cmake_minimum_required(VERSION 3.16)
project(kvmem_llamacpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

enable_testing()
add_subdirectory(kvmem)

option(KVMEM_BUILD_LLAMA "Build llama.cpp with the KVMem adapter" ON)
if (KVMEM_BUILD_LLAMA)
    if (NOT CMAKE_CUDA_COMPILER AND EXISTS "/home/leye/kvmem_qw3/.cu13-env/bin/nvcc")
        set(CMAKE_CUDA_COMPILER "/home/leye/kvmem_qw3/.cu13-env/bin/nvcc" CACHE FILEPATH "nvcc")
    endif()
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "120a-real" CACHE STRING "Blackwell sm_120")
    endif()

    option(GGML_CUDA "ggml: use CUDA" ON)
    set(LLAMA_KVMEM ON CACHE BOOL "" FORCE)
    set(LLAMA_KVMEM_ROOT "${CMAKE_SOURCE_DIR}" CACHE PATH "" FORCE)
    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TOOLS ON CACHE BOOL "")
    set(LLAMA_BUILD_COMMON ON CACHE BOOL "")

    add_subdirectory(llama.cpp)

    add_executable(llama-kvmem-cli tools/llama-kvmem-cli.cpp tools/kvmem-spec.cpp)
    target_include_directories(llama-kvmem-cli PRIVATE
        src/adapter
        llama.cpp/include
        llama.cpp/common)
    target_link_libraries(llama-kvmem-cli PRIVATE llama llama-common)
    target_compile_features(llama-kvmem-cli PRIVATE cxx_std_17)

    add_executable(llama-kvmem-server tools/llama-kvmem-server.cpp tools/kvmem-spec.cpp)
    target_include_directories(llama-kvmem-server PRIVATE
        src/adapter
        llama.cpp/include
        llama.cpp/common
        llama.cpp/vendor
        llama.cpp/vendor/cpp-httplib)
    target_link_libraries(llama-kvmem-server PRIVATE llama llama-common cpp-httplib)
    target_compile_features(llama-kvmem-server PRIVATE cxx_std_17)
endif()
