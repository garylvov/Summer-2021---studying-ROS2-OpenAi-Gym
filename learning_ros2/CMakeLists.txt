cmake_minimum_required(VERSION 3.8)
project(learning_ros2)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(turtlesim REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(geometry2 REQUIRED)
find_package(tf2_ros REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
include_directories(include)
# Create Cpp executable
add_executable(cpp_executable src/cpp_node.cpp)
ament_target_dependencies(cpp_executable rclcpp)

add_executable(go_to_goal_turtle src/turtlesim/go_to_goal_turtle.cpp)
ament_target_dependencies(go_to_goal_turtle rclcpp turtlesim geometry_msgs)

add_executable(tf_broadcaster src/turtlesim/tf_broadcaster.cpp)
ament_target_dependencies(tf_broadcaster rclcpp turtlesim geometry_msgs geometry2 tf2_ros)

add_executable(go_to_goal_turtle_tf src/turtlesim/go_to_goal_turtle_tf.cpp)
ament_target_dependencies(go_to_goal_turtle_tf rclcpp turtlesim geometry_msgs geometry2 tf2_ros)
# Install Cpp executables
install(TARGETS
  cpp_executable
  go_to_goal_turtle
  tf_broadcaster
  go_to_goal_turtle_tf
  DESTINATION lib/${PROJECT_NAME}
)
# Install Python modules
ament_python_install_package(${PROJECT_NAME})
# Install Python executables
install(PROGRAMS
  scripts/py_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

ament_package()