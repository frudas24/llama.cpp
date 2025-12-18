set(TEST_TARGET llama-seeddelta-selftest)
add_executable(${TEST_TARGET}
    sd_selftest.cpp
)
target_link_libraries(${TEST_TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
