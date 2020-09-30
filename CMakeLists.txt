SET(PROJECT_NAME open_quadtree_mapping)
CMAKE_MINIMUM_REQUIRED(VERSION 3.17)
PROJECT(${PROJECT_NAME} LANGUAGES CUDA CXX)
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/CMakeModules/")

find_package(CUDA REQUIRED)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -use_fast_math)

# Specify compute capability
# NVIDIA Quardo T1000 compute capacity : 7.5
list(APPEND CUDA_NVCC_FLAGS -arch=sm_75)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_75,code=sm_75)

SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenCV REQUIRED)

find_package(Eigen REQUIRED)

find_package(Boost REQUIRED)

include_directories(
        include
        ${OpenCV_INCLUDE_DIRS}
        ${Eigen_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
)

# Set link libraries
list(APPEND LINK_LIBS
        ${OpenCV_LIBS}
        ${Boost_LIBRARIES}
        )

add_executable(open_quadtree_mapping_node
        src/check_cuda_device.cu
        src/seed_matrix.cu
        src/depthmap.cpp
        src/main_cu.cpp
        )
target_link_libraries(open_quadtree_mapping_node
         ${OpenCV_LIBS} ${Boost_LIBRARIES})