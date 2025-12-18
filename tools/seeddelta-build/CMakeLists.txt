set(TARGET llama-seeddelta-build)
add_executable(${TARGET}
    seeddelta-build.cpp
    seeddelta_policy.cpp
    seeddelta_policy_export.cpp
    seeddelta_policy_selftest.cpp
    sd_eval.cpp
    sd_cost.cpp
    sd_ffn_proxy.cpp
    sd_report.cpp
)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()

add_subdirectory(selftest)
