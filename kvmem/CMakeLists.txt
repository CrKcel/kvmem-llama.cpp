add_library(kvmem
    src/host/kvmem_store.cpp
    src/host/kvmem_runtime.cpp
    src/host/raw_kv_store.cpp
    src/host/rope.cpp
)
set_target_properties(kvmem PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(kvmem PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(kvmem PUBLIC cxx_std_17)

function(kvmem_add_test name)
    add_executable(${name} tests/${name}.cpp)
    target_link_libraries(${name} PRIVATE kvmem)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

kvmem_add_test(kvmem_store_test)
kvmem_add_test(pinned_kv_tier_test)
kvmem_add_test(nvme_kv_tier_test)
kvmem_add_test(kvmem_runtime_test)
kvmem_add_test(raw_kv_store_test)
