/*
Copyright 2021 Gary Lvov
 
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
 
 http://www.apache.org/licenses/LICENSE-2.0
 
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
 
garylvov.com
 
Sources Consulted:
http://wiki.ros.org/tf/Tutorials/Writing%20a%20tf%20broadcaster%20%28C%2B%2B%29
https://answers.ros.org/question/360516/ros2-tf2-broadcaster/
https://github.com/garylvov/learning_ROS-Gazebo_summer_2020/blob/main/learning_tf2/nodes/bcOptimized.py
*/
#include "rclcpp/rclcpp.hpp"
#include <turtlesim/msg/pose.hpp>
#include <turtlesim/srv/spawn.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>

class TFBroadcaster : public rclcpp::Node
{
 public:
   TFBroadcaster() : Node("TF_broadcaster")
   {
     my_pose_sub_ = this->create_subscription<turtlesim::msg::Pose>(
       "/turtle1/pose", 10, std::bind(&TFBroadcaster::my_pose_callback, this, std::placeholders::_1));
     goal_pose_sub_ = this->create_subscription<turtlesim::msg::Pose>(
       "/goal_turtle/pose", 10, std::bind(&TFBroadcaster::goal_pose_callback, this, std::placeholders::_1));

     tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
    }
 private:
  geometry_msgs::msg::TransformStamped turtle1_transform;
  geometry_msgs::msg::TransformStamped goal_turtle_transform;
  tf2::Quaternion turtle1_q;
  tf2::Quaternion goal_turtle_q;

   void my_pose_callback(const turtlesim::msg::Pose::SharedPtr msg)
   {
    create_tranform("world","turtle1", turtle1_transform, turtle1_q, msg);
   }
   void goal_pose_callback(const turtlesim::msg::Pose::SharedPtr msg)
   {
    create_tranform("world","goal_turtle", goal_turtle_transform, goal_turtle_q, msg); 
   }
   void create_tranform(std::string parent, std::string child, geometry_msgs::msg::TransformStamped transform_stamped,
                        tf2::Quaternion q, const turtlesim::msg::Pose::SharedPtr msg)
    {
     transform_stamped.header.stamp = rclcpp::Time();
     transform_stamped.header.frame_id = parent;
     transform_stamped.child_frame_id = child;
     transform_stamped.transform.translation.x = msg->x;
     transform_stamped.transform.translation.y = msg->y;
     transform_stamped.transform.translation.z = 0.0;
     q.setRPY(0, 0, msg ->theta);
     transform_stamped.transform.rotation.x = q.x();
     transform_stamped.transform.rotation.y = q.y();
     transform_stamped.transform.rotation.z = q.z();
     transform_stamped.transform.rotation.w = q.w();
   }
   rclcpp::Subscription<turtlesim::msg::Pose>::SharedPtr my_pose_sub_;
   rclcpp::Subscription<turtlesim::msg::Pose>::SharedPtr goal_pose_sub_;
   std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};
int main(int argc, char *argv[])
{
 rclcpp::init(argc, argv);
 rclcpp::spin(std::make_shared<TFBroadcaster>());
 rclcpp::shutdown();
 return 0;
}
