cmake_minimum_required(VERSION 3.23)
project(ParticleFilter)

set(CMAKE_CXX_STANDARD 17)
find_package(Eigen3 REQUIRED)
include_directories(include)
set(SRC_FILES
        src/particle_filter.cpp include/particle_filter.h include/filter_base.h src/filter_base.cpp
        )
add_executable(ParticleFilter main.cpp ${SRC_FILES})
target_link_libraries(${PROJECT_NAME} Eigen3::Eigen)
