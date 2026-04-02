#!/usr/bin/env python3
import sys
import threading
import time
import termios
import tty
import serial
import select

import rclpy
from rclpy.node import Node


KEYMAP = {
    'z': "D 50 50 1",     # forward
    'a': "D 50 -50 1",    # left
    'e': "D -50 50 1",    # right
    's': "D -50 -50 1",   # backward
}

HELP = """\
Controls:
  z = forward        -> D 50 50 1
  a = left           -> D 50 -50 1
  e = right          -> D -50 50 1
  s = backward       -> D -50 -50
  q = quit

Focus the terminal window and press keys (no Enter needed).
"""

class SerialTeleop(Node):
    def __init__(self):
        super().__init__('serial_teleop')

        # Parameters (with your requested defaults)
        self.declare_parameter('port', '/dev/ttyACM0')
        self.declare_parameter('baud', 57600)
        self.declare_parameter('timeout', 0.1)   # serial read timeout (s)

        port = self.get_parameter('port').get_parameter_value().string_value
        baud = self.get_parameter('baud').get_parameter_value().integer_value
        timeout = self.get_parameter('timeout').get_parameter_value().double_value

        # Open serial
        try:
            self.ser = serial.Serial(port, baudrate=baud, timeout=timeout)
        except Exception as e:
            self.get_logger().error(f"Failed to open serial {port} @ {baud}: {e}")
            raise

        # Let MCU (e.g., Arduino) reset if needed
        time.sleep(2.0)
        self.get_logger().info(f"Connected to {port} @ {baud}")
        self.get_logger().info("\n" + HELP)

        # Start background thread for reading serial responses
        self._stop_event = threading.Event()
        self.reader_thread = threading.Thread(target=self._serial_reader_loop, daemon=True)
        self.reader_thread.start()

        # Setup terminal raw mode to read single keys
        self._orig_term = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())

        # Use a timer to poll stdin without blocking ROS spin
        self.key_timer = self.create_timer(0.02, self._poll_keyboard)

    def _serial_reader_loop(self):
        """Continuously read lines from serial and print them."""
        while not self._stop_event.is_set():
            try:
                if self.ser.in_waiting:
                    line = self.ser.readline().decode(errors='ignore').strip()
                    if line:
                        self.get_logger().info(f"< {line}")
                else:
                    # small sleep to avoid busy loop
                    time.sleep(0.01)
            except Exception as e:
                self.get_logger().warn(f"Serial read error: {e}")
                time.sleep(0.1)

    def _poll_keyboard(self):
        """Non-blocking single-key read from stdin; map to serial commands."""
        try:
            rlist, _, _ = select.select([sys.stdin], [], [], 0)
            if rlist:
                ch = sys.stdin.read(1)
                if not ch:
                    return

                if ch == 'q':
                    self.get_logger().info("Quit requested.")
                    # shutdown node
                    rclpy.shutdown()
                    return

                if ch in KEYMAP:
                    cmd = KEYMAP[ch]
                    try:
                        self.ser.write((cmd + "\n").encode())
                        self.get_logger().info(f"> {cmd}")
                    except Exception as e:
                        self.get_logger().error(f"Serial write failed: {e}")
                elif ch == 'h':
                    self.get_logger().info("\n" + HELP)
                else:
                    # ignore unknown keys but give a tiny hint on '?'
                    if ch == '?':
                        self.get_logger().info("\n" + HELP)

        except Exception as e:
            self.get_logger().warn(f"Keyboard read error: {e}")

    def destroy_node(self):
        # Restore terminal
        try:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._orig_term)
        except Exception:
            pass

        # Stop threads and close serial
        self._stop_event.set()
        try:
            if self.reader_thread.is_alive():
                self.reader_thread.join(timeout=0.5)
        except Exception:
            pass

        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass

        super().destroy_node()


def main():
    rclpy.init()
    node = SerialTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()
