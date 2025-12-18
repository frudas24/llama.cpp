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
target_compile_features(${TARGET} PRIVATE cxx_std_17)

target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()

add_executable(llama-seeddelta-report-selftest
    selftest/report_selftest.cpp
)
target_include_directories(llama-seeddelta-report-selftest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(llama-seeddelta-report-selftest PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(llama-seeddelta-report-selftest PRIVATE cxx_std_17)
