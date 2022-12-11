cmake_minimum_required(VERSION 3.15)

PROJECT(demo)
find_package(libufget)
add_executable(demo demo.c)
target_link_libraries(demo PRIVATE libufget::ufget)

