set(TEST_TARGET llama-seeddelta-selftest)
add_executable(${TEST_TARGET}
    sd_selftest.cpp
    ../seeddelta_policy.cpp
    ../sd_policy_resolve.cpp
    ../sd_report.cpp
)
target_link_libraries(${TEST_TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TEST_TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../include)

set(TOY_TARGET llama-seeddelta-toy)
add_executable(${TOY_TARGET}
    sd_toy_matrix.cpp
    ../sd_eval.cpp
)
target_link_libraries(${TOY_TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TOY_TARGET} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../include)
