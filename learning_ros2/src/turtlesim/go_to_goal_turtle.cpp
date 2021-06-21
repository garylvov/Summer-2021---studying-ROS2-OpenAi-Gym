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

Sources consulted:
https://docs.ros.org/en/galactic/Tutorials/Writing-A-Simple-Cpp-Service-And-Client.html
https://docs.ros.org/en/galactic/Tutorials/Writing-A-Simple-Cpp-Publisher-And-Subscriber.html
https://github.com/ros/ros_tutorials/blob/galactic-devel/turtlesim/tutorials/mimic.cpp
https://github.com/garylvov/learning_ROS-Gazebo_summer_2020/blob/main/starting_turtle_sim/goToGoalTurtleV3.py
*/
#include <rclcpp/rclcpp.hpp>
#include <cmath>

#include <unistd.h>
#include <geometry_msgs/msg/twist.hpp>
#include <turtlesim/msg/pose.hpp>
#include <turtlesim/srv/kill.hpp>
#include <turtlesim/srv/spawn.hpp>

using namespace std::chrono_literals;

class Turtle_handler : public rclcpp::Node
{       
public:
    Turtle_handler() : rclcpp::Node("turtle_handler_tf")
    {
        my_pose_sub_ = this->create_subscription<turtlesim::msg::Pose>(
      "/turtle1/pose", 10, std::bind(&Turtle_handler::my_pose_callback, this, std::placeholders::_1));

        goal_pose_sub_ = this->create_subscription<turtlesim::msg::Pose>(
      "/goal_turtle/pose", 10, std::bind(&Turtle_handler::goal_pose_callback, this, std::placeholders::_1));

        my_twist_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("turtle1/cmd_vel", 1);

        timer_ = this-> create_wall_timer(100ms, std::bind(&Turtle_handler::get_path, this)); //10 Hz

        spawn_client_ = this->create_client<turtlesim::srv::Spawn>("spawn");
        kill_client_ = this->create_client<turtlesim::srv::Kill>("kill");

        spawn_turtle_random_coords(); //Creates first turtle
    }

private:
    turtlesim::msg::Pose my_pose;
    turtlesim::msg::Pose goal_pose;
    geometry_msgs::msg::Twist my_twist; 

    void my_pose_callback(const turtlesim::msg::Pose& msg)
    {   
      my_pose = msg;
    }
    void goal_pose_callback(const turtlesim::msg::Pose& msg){
      goal_pose = msg;
    }
    void get_path(/*turtlesim::msg::Pose my_turtle, turtlesim::msg::Pose goal_turtle*/)
    {   
        double x = (goal_pose.x) - (my_pose.x);
        double y = (goal_pose.y) - (my_pose.y);
        double dist = sqrt(pow(x,2) + (pow(y,2)));
        double heading = (atan2(y,x)) - (my_pose.theta);

        float lin = dist;
        float ang = heading * 1.5;

        if(abs(heading) > .05)
        {
            lin = dist/10; ang = heading * 4;
            if(abs(heading) > M_PI){ang = (M_PI - heading) * 4;}
        }

        else{lin = dist*3; ang = 0;}

        if(dist < .5){reset();}

        my_twist.linear.x = lin;
        my_twist.angular.z = ang;
        my_twist_pub_->publish(my_twist);
    }
    void spawn_turtle_random_coords()
    { 
        srand(time(0));
        double x = ((rand() % 1000) * .001) + rand() %10+1;
        double y = ((rand() % 1000) * .001) + rand() %10+1;
        double theta = ((rand() % 1000) * .001) + rand() % 3;

        auto spawn_request = std::make_shared<turtlesim::srv::Spawn::Request>();
        spawn_request -> name = "goal_turtle";
        spawn_request -> x = x;
        spawn_request -> y = y;
        spawn_request -> theta = theta;

        auto result = spawn_client_->async_send_request(spawn_request);
    }
    void kill_turtle()
    {
        auto kill_request = std::make_shared<turtlesim::srv::Kill::Request>();
        kill_request -> name = "goal_turtle";
        auto result = kill_client_->async_send_request(kill_request);
        RCLCPP_INFO(this->get_logger(), "Yum!");
    }
    void reset(){kill_turtle(); usleep(20000) /*TODO: replace usleep with check that Kill ran*/; spawn_turtle_random_coords();}

    rclcpp::Subscription<turtlesim::msg::Pose>::SharedPtr my_pose_sub_; 
    rclcpp::Subscription<turtlesim::msg::Pose>::SharedPtr goal_pose_sub_; 
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr my_twist_pub_;

    rclcpp::TimerBase::SharedPtr timer_;

    rclcpp::Client<turtlesim::srv::Spawn>::SharedPtr spawn_client_;
    rclcpp::Client<turtlesim::srv::Kill>::SharedPtr kill_client_;
    
};
int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Turtle_handler>());
  rclcpp::shutdown();
  return 0;
}
