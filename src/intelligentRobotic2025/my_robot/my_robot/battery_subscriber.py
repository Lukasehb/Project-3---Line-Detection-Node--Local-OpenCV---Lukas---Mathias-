import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

class BatterySubscriber(Node):
    def __init__(self):
        super().__init__('battery_sub')
        self.sub = self.create_subscription(Float32, 'battery_voltage', self.cb, 10)

    def cb(self, msg):
        v = msg.data
        self.get_logger().info(str(v))
        if v < 17.5:
            self.get_logger().warn('LOW BATTERY')

def main(args=None):
    rclpy.init(args=args)
    node = BatterySubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

