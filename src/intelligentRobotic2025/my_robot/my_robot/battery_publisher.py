
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import random

class BatteryPublisher(Node):
    def __init__(self):
        super().__init__('battery_pub')
        self.pub = self.create_publisher(Float32, 'battery_voltage', 10)
        self.timer = self.create_timer(60.0, self.timer_cb)

    def timer_cb(self):
        msg = Float32()
        msg.data = random.uniform(10, 13)
        self.pub.publish(msg)
        self.get_logger().info(str(msg.data))

def main(args=None):
    rclpy.init(args=args)
    node = BatteryPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
