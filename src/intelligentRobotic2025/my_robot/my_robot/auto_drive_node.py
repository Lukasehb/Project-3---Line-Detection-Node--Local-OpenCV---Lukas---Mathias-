import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import serial
import time

class AutoDriveNode(Node):
    def __init__(self):
        super().__init__('auto_drive_node')
        
        try:
            self.serial_port = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
            self.get_logger().info("Serial port: OPEN")
        except serial.SerialException as e:
            self.get_logger().error(f"Serial port: FAILED - {e}")
            self.serial_port = None

        self.subscription = self.create_subscription(Twist, '/cmd_vel', self.listener_callback, 10)

        # 2 Hz Throttle Limit
        self.last_send_time = 0.0
        self.throttle_interval = 0.5 

        self.latest_linear = 0.0
        self.latest_angular = 0.0

    def listener_callback(self, msg):
        self.latest_linear = msg.linear.x
        self.latest_angular = msg.angular.z
        
        current_time = time.time()
        if current_time - self.last_send_time >= self.throttle_interval:
            self.send_to_motors()
            self.last_send_time = current_time

    def send_to_motors(self):
        if self.serial_port is None:
            return

        # Differential drive mix
        left_speed = int(self.latest_linear - self.latest_angular)
        right_speed = int(self.latest_linear + self.latest_angular)

        # Hard bounds
        left_speed = max(min(left_speed, 10), -10)
        right_speed = max(min(right_speed, 10), -10)

        command_str = f"{left_speed},{right_speed}\n"
        self.serial_port.write(command_str.encode('utf-8'))
        self.get_logger().info(f"MOTORS: {command_str.strip()}")

def main(args=None):
    rclpy.init(args=args)
    auto_drive = AutoDriveNode()
    try:
        rclpy.spin(auto_drive)
    except KeyboardInterrupt:
        pass
    finally:
        if auto_drive.serial_port:
            auto_drive.serial_port.write(b"0,0\n")
            auto_drive.serial_port.close()
        auto_drive.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()