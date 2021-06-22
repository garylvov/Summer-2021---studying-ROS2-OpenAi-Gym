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
http://wiki.ros.org/tf2/Tutorials/Writing%20a%20tf2%20listener%20%28C%2B%2B%29
https://answers.ros.org/question/338515/ros2are-there-any-examples-of-tf2-broadcast-or-listener/
https://github.com/ros2/geometry2/blob/ros2/tf2_ros/src/tf2_echo.cpp
https://github.com/garylvov/learning_ROS2-MoveIt-ML_summer_2021/blob/main/learning_ros2/src/turtlesim/go_to_goal_turtle.cpp
*/
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/time_source.hpp>
#include <cmath>
#include <unistd.h>
#include <turtlesim/msg/pose.hpp>
#include <turtlesim/srv/spawn.hpp>
#include <turtlesim/srv/kill.hpp>
#include <turtlesim/srv/set_pen.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

using namespace std::chrono_literals;

class Turtle_handler : public rclcpp::Node
{      
public:
    Turtle_handler() : rclcpp::Node("turtle_handler")
    {
        my_twist_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("turtle1/cmd_vel", 1);

        buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        transform_listener_ = std::make_shared<tf2_ros::TransformListener>(*buffer_);

        timer_ = this-> create_wall_timer(100ms, std::bind(&Turtle_handler::get_path, this)); //10 Hz

        spawn_client_ = this->create_client<turtlesim::srv::Spawn>("spawn");
        kill_client_ = this->create_client<turtlesim::srv::Kill>("kill");
        set_pen_client_ = this->create_client<turtlesim::srv::SetPen>("turtle1/set_pen");
    }

private:
    geometry_msgs::msg::Twist my_twist; 
    geometry_msgs::msg::TransformStamped transformStamped;

    void get_path()
    {     
        try {

            transformStamped = buffer_->lookupTransform("turtle1", "goal_turtle", tf2::TimePointZero);
        }
        catch (tf2::TransformException & ex) {
            RCLCPP_ERROR(this->get_logger(), "StaticLayer: %s", ex.what());
        }
        double x = transformStamped.transform.translation.x;
        double y = transformStamped.transform.translation.y;
        double dist = sqrt(pow(x,2) + (pow(y,2)));
        double heading = (atan2(y,x));

        float lin = dist * 2.25;
        float ang = heading * 4.25;

        if(dist < .4){reset();}

        my_twist.linear.x = lin;
        my_twist.angular.z = ang; 
        my_twist_pub_->publish(my_twist);
    }
    void spawn_turtle_random_coords()
    { 
        srand(time(0));
        double x = ((rand() % 1000) * .001) + rand() %9 +1;
        double y = ((rand() % 1000) * .001) + rand() %9 +1;
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

    void set_pen_random()
    { 
        srand(time(0));
        int r = rand() %250;
        int g = rand() %250;
        int b = rand() %250;
        int width = 10; //rand() %50;

        auto set_pen_request = std::make_shared<turtlesim::srv::SetPen::Request>();
        set_pen_request -> r = r;
        set_pen_request -> g = g;
        set_pen_request -> b = b;
        set_pen_request -> width = width;
        auto result = set_pen_client_->async_send_request(set_pen_request);
    }
    void reset()
    {
        kill_turtle(); 
        usleep(20000);  /*TODO: replace usleep with check that Kill ran*/
        set_pen_random();
        spawn_turtle_random_coords();
    }
 
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr my_twist_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    
    std::unique_ptr<tf2_ros::Buffer> buffer_;
    std::shared_ptr<tf2_ros::TransformListener> transform_listener_;

    rclcpp::Client<turtlesim::srv::Spawn>::SharedPtr spawn_client_;
    rclcpp::Client<turtlesim::srv::Kill>::SharedPtr kill_client_;
    rclcpp::Client<turtlesim::srv::SetPen>::SharedPtr set_pen_client_;
};
int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Turtle_handler>());
  rclcpp::shutdown();
  return 0;
}
