#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

using namespace std::chrono_literals;

//Minimal Publisher inherits from rclcpp::Node 
class MinimalPublisher : public rclcpp::Node
{
  //declaration of timer, publisher, and counter
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
  size_t count_;

  public:
  //sets count to 0
    MinimalPublisher() 
    : Node("minimal_publisher"), count_(0)
    {
      //creates publisher object - String type, under the name topic
      publisher_ = this->create_publisher<std_msgs::msg::String>("topic", 10);

      //creates callback function that runs twice a second
      timer_ = this-> create_wall_timer(500ms, std::bind(&MinimalPublisher::timer_callback, this));
    }

  private:
    //this actually sets/publishes the message data
    void timer_callback()
    {
      auto message = std_msgs::msg::String();
      message.data = "Hello, world! " + std::to_string(count_++);

      //Prints to console
      RCLCPP_INFO(this->get_logger(), "Publishing : '%s'", message.data.c_str());
      publisher_->publish(message);
    }
};

int main(int argc, char *argv[])
{
  //initaites nodes, starts 
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MinimalPublisher>());
  rclcpp::shutdown();
  return 0;
}
