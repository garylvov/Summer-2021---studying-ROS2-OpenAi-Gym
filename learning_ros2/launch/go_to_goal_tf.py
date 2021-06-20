from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
         Node(
            package='turtlesim',
            executable='turtlesim_node'
        ),
        Node(
            package='learning_ros2',
            executable='tf_broadcaster'
        ),
        Node(
                package='learning_ros2',
                executable='go_to_goal_turtle_tf'
            )
    ])
