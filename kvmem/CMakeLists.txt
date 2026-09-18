if (WIN32)
    set(KVMEM_NVME_DEFAULT OFF)
else()
    set(KVMEM_NVME_DEFAULT ON)
endif()
option(KVMEM_ENABLE_NVME "Build the POSIX NVMe KV storage tier" ${KVMEM_NVME_DEFAULT})
if (WIN32 AND KVMEM_ENABLE_NVME)
    message(FATAL_ERROR "NVMe storage is not supported on Windows; use KVMEM_ENABLE_NVME=OFF")
endif()

add_library(kvmem
    src/host/kvmem_store.cpp
    src/host/kvmem_runtime.cpp
    src/host/raw_kv_store.cpp
    src/host/rope.cpp
)
set_target_properties(kvmem PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(kvmem PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(kvmem PUBLIC cxx_std_17)
target_compile_definitions(kvmem PUBLIC KVMEM_ENABLE_NVME=$<BOOL:${KVMEM_ENABLE_NVME}>)

function(kvmem_add_test name)
    add_executable(${name} tests/${name}.cpp)
    target_link_libraries(${name} PRIVATE kvmem)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

kvmem_add_test(kvmem_store_test)
kvmem_add_test(pinned_kv_tier_test)
if (KVMEM_ENABLE_NVME)
    kvmem_add_test(nvme_kv_tier_test)
else()
    kvmem_add_test(nvme_disabled_test)
endif()
kvmem_add_test(kvmem_runtime_test)
kvmem_add_test(raw_kv_store_test)
