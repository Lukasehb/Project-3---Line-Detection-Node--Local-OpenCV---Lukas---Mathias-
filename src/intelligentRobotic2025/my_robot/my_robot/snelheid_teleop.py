#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import serial

class SnelheidTeleop(Node):
    def __init__(self):
        super().__init__("snelheid_teleop")

        # seriële poort openen
        self.ser = serial.Serial("/dev/ttyACM0", 57600, timeout=0.1)

        # snelheden
        self.l = 0
        self.r = 0

        # Info bij opstart
        self.get_logger().info("Snelheid Teleop gestart!")
        self.get_logger().info("-------------------------------------------")
        self.get_logger().info("Toetsenbord commando's:")
        self.get_logger().info("  z -> sneller vooruit (+5)")
        self.get_logger().info("  s -> trager / achteruit (-5)")
        self.get_logger().info("  a -> links draaien (+5 / -5)")
        self.get_logger().info("  e -> rechts draaien (-5 / +5)")
        self.get_logger().info("  x -> STOP (0 / 0)")
        self.get_logger().info("  q -> stoppen")
        self.get_logger().info("-------------------------------------------")
        self.get_logger().info(f"Start snelheid: L = {self.l} | R = {self.r}")
        self.get_logger().info("-------------------------------------------")

        self.loop()

    def stuur(self):
        cmd = f"V {self.l} {self.r}"
        self.ser.write((cmd + "\n").encode())
        self.get_logger().info(f"Verstuurd: {cmd} (L={self.l}, R={self.r})")

    def loop(self):
        while rclpy.ok():
            key = input("toets: ").strip()

            if key == "q":
                self.get_logger().info("Programma afgesloten.")
                break

            if key == "z":
                self.l += 5
                self.r += 5
                self.get_logger().info("Actie: sneller vooruit (+5)")

            elif key == "s":
                self.l -= 5
                self.r -= 5
                self.get_logger().info("Actie: trager / achteruit (-5)")

            elif key == "a":
                self.l += 5
                self.r -= 5
                self.get_logger().info("Actie: links draaien")

            elif key == "e":
                self.l -= 5
                self.r += 5
                self.get_logger().info("Actie: rechts draaien")

            elif key == "x":
                self.l = 0
                self.r = 0
                self.get_logger().info("Actie: STOP")

            else:
                self.get_logger().info("Ongeldige toets.")
                continue

            # snelheid naar robot sturen
            self.stuur()


def main():
    rclpy.init()
    node = SnelheidTeleop()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
