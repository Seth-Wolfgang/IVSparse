# To build, type: cmake -B /path/to/folder/you/want/it/built/in -S .
# Or else you will have 4 new directories and 3 new files in source
# Makefile will be in the build folder
cmake_minimum_required(VERSION 3.14)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/tests/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/tests/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/tests/)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/tests)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")



project(IVCSC)
# add_executable(main ${CMAKE_SOURCE_DIR}/tests/testIVCSC.cpp)
enable_testing()

# Required for googletest
set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(
  testingMain   
  ${CMAKE_SOURCE_DIR}/tests/testingMain.cpp
)

target_link_libraries(
  testingMain
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(testingMain)