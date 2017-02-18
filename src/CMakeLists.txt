cmake_minimum_required(VERSION 2.8.11)
include_directories(${PROJECT_SOURCE_DIR}/3pp/spdlog/include)
file(GLOB_RECURSE Sources "*.cpp" "*.h")
add_executable(${PROJECT_NAME} ${Sources})

# Qt library
find_package(Qt5Widgets)

target_link_libraries(${PROJECT_NAME} ${CMAKE_THREAD_LIBS_INIT} Qt5::Widgets)



