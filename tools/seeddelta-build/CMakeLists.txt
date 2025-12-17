set(TARGET llama-seeddelta-build)
add_executable(${TARGET}
    seeddelta-build.cpp
    seeddelta_policy.cpp
    seeddelta_policy_export.cpp
    seeddelta_policy_selftest.cpp
)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
