cmake_minimum_required(VERSION 4.0.0)
project(HelloSAM VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 CONFIG REQUIRED)
find_package(Sophus CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(ceres CONFIG REQUIRED)

add_executable(main main.cpp)

target_link_libraries(main PRIVATE
    Eigen3::Eigen
    Sophus::Sophus
    ${OpenCV_LIBS}
    Ceres::ceres
)