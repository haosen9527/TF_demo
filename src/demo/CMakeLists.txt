cmake_minimum_required(VERSION 2.8.3)
project(demo)
add_compile_options(-std=c++11)
find_package(catkin REQUIRED COMPONENTS
  roscpp
)

include_directories( "/usr/local/include/eigen3" )

include_directories(
  #include
  "~/tensorflow"
  "~/tensorflow/bazel-genfiles"
  "~/tensorflow/tensorflow/contrib/makefile/gen/protobuf/include"
  ${catkin_INCLUDE_DIRS}
)
link_directories("~/tensorflow/bazel-bin/tensorflow")

include_directories(
  ${catkin_INCLUDE_DIRS}
  ~/tensorflow_cxx/tensorflow_source/tensorflow
  ~/tensorflow_cxx/tensorflow_source/tensorflow/bazel-genfiles
  ~/tensorflow_cxx/tensorflow_source/tensorflow/tensorflow/contrib/makefile/gen/protobuf/include
)
link_directories(~/tensorflow_cxx/tensorflow_source/tensorflow/bazel-bin/tensorflow)


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES example_traner
#  CATKIN_DEPENDS roscpp rospy
#  DEPENDS system_lib
)

add_executable(demo_test  src/demo.cpp)
target_link_libraries(demo_test
    tensorflow_cc
    tensorflow_framework
    )

add_executable(crate_graph  src/crate_graph.cpp)
target_link_libraries(crate_graph
    tensorflow_cc
    tensorflow_framework
    ${catkin_LIBRARIES}
    )
#Record_demo.cpp
add_executable(Record_demo  src/Record_demo.cpp)
target_link_libraries(Record_demo
    tensorflow_cc
    tensorflow_framework
    ${catkin_LIBRARIES}
    )
#graph_logging.cpp
add_executable(graph_logging  src/graph_logging.cpp)
target_link_libraries(graph_logging
    tensorflow_cc
    tensorflow_framework
    ${catkin_LIBRARIES}
    )
#mnist_cxx.cpp
add_executable(mnist_cxx  src/mnist_cxx.cpp)
target_link_libraries(mnist_cxx
    tensorflow_cc
    tensorflow_framework
    ${catkin_LIBRARIES}
    )

#line_test.cpp
add_executable(line_test  src/line_test.cpp)
target_link_libraries(line_test
    tensorflow_cc
    tensorflow_framework
    ${catkin_LIBRARIES}
    )
#tensor_image.cpp
add_executable(tensor_image  src/tensor_image.cpp)
target_link_libraries(tensor_image
    tensorflow_cc
    tensorflow_framework
    ${catkin_LIBRARIES}
    )
#Network.cpp
add_executable(network  src/network.cpp)
target_link_libraries(network
    tensorflow_cc
    tensorflow_framework
    ${catkin_LIBRARIES}
    )
#vggnet_test.cpp
#add_executable(vggnet_test src/vggnet_test.cpp)
#target_link_libraries(network
#    tensorflow_cc
#    tensorflow_framework
#    ${catkin_LIBRARIES}
#    )


