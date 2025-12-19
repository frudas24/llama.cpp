set(TARGET llama-statecells-build)
add_executable(${TARGET}
    statecells-build.cpp
    sc_cli.cpp
    sc_build.cpp
    sc_runner.cpp
    sc_imatrix.cpp
    sc_encode_train.cpp
    sc_encode_codes.cpp
    sc_encode_row_scale.cpp
    sc_eval.cpp
    sc_utils.cpp
    sc_process.cpp
    sc_report.cpp
)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
