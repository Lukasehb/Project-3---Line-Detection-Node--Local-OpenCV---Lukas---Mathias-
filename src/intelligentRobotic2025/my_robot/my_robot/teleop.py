#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import serial

class SerialTeleop(Node):
    def __init__(self):
        super().__init__("serial_teleop")

        # serial openen
        self.ser = serial.Serial("/dev/ttyACM0", 57600, timeout=0.1)
        self.get_logger().info("Connected to /dev/ttyACM0 @ 57600")
        self.get_logger().info("Use keys: z a e s (q = quit)")

        self.main_loop()

    def main_loop(self):
        while rclpy.ok():
            key = input("key: ").strip()

            if key == "q":
                self.get_logger().info("Quit.")
                break

            if key == "z":
                cmd = "D 50 50 1"
            elif key == "s":
                cmd = "D -50 -50 1"
            elif key == "a":
                cmd = "D 50 -50 1"
            elif key == "e":
                cmd = "D -50 50 1"
            else:
                self.get_logger().info("Invalid key.")
                continue

            self.ser.write((cmd + "\n").encode())
            self.get_logger().info(f"Sent: {cmd}")

def main():
    rclpy.init()
    node = SerialTeleop()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
