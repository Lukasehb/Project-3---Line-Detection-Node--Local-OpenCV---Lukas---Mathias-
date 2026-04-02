#!/usr/bin/env python3
import os
import time
import cv2
import numpy as np
import rclpy
import serial
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32

class LineDetectorNode(Node):
    def __init__(self):
        super().__init__('line_detector_node')

        self.declare_parameter('device',       '/dev/video0')
        self.declare_parameter('width',        640)
        self.declare_parameter('height',       480)
        self.declare_parameter('fps',          15.0)
        self.declare_parameter('topic',        '/camera/image_raw')
        self.declare_parameter('frame_id',     'camera_frame')
        self.declare_parameter('port',         '/dev/ttyACM0')
        self.declare_parameter('baud',         57600)
        self.declare_parameter('kp',           0.35)
        self.declare_parameter('kd',           0.15)
        self.declare_parameter('base_pwm',     50)
        self.declare_parameter('line_target_x', 320)
        self.declare_parameter('serial_delay', 0.05)
        self.declare_parameter('record',       False)
        self.declare_parameter('record_path',  '~/robot_debug.avi')

        self.prev_error = 0
        self.last_error_time = time.time()
        self.last_serial_time = time.time()
        self.smoothed_lx = None

        port = str(self.get_parameter('port').value)
        baud = int(self.get_parameter('baud').value)
        self.ser = None
        try:
            self.ser = serial.Serial(port, baudrate=baud, timeout=0.1)
            time.sleep(2.0)
        except Exception:
            pass

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.pub_raw = self.create_publisher(Image, str(self.get_parameter('topic').value), qos)
        self.pub_debug = self.create_publisher(Image, '/line_detector/debug_image', qos)
        self.pub_heading = self.create_publisher(Float32, '/line_heading', 10)

        dev = str(self.get_parameter('device').value)
        self.cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  int(self.get_parameter('width').value))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.get_parameter('height').value))

        self.writer = None
        if self.get_parameter('record').value:
            path = os.path.expanduser(str(self.get_parameter('record_path').value))
            self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'XVID'), 15.0, (640, 480))

        self.timer = self.create_timer(1.0/15.0, self._tick)

    def _send_serial_cmd(self, left_pwm, right_pwm):
        if not self.ser or not self.ser.is_open: return
        try:
            self.ser.write(f"D {int(left_pwm)} {int(right_pwm)} 1\n".encode())
            self.ser.flush()
        except Exception: pass

    def _stop_motors(self):
        for _ in range(3):
            self._send_serial_cmd(0, 0)
            time.sleep(0.02)

    def _get_largest_contour_center(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < 50: return None
        M = cv2.moments(largest_contour)
        if M['m00'] == 0: return None
        return int(M['m10'] / M['m00'])

    def _tick(self):
        ok, frame = self.cap.read()
        if not ok or frame is None: return
        
        h, w = frame.shape[:2]
        now = time.time()
        frame_id = str(self.get_parameter('frame_id').value)

        msg_raw = Image()
        msg_raw.header.stamp = self.get_clock().now().to_msg()
        msg_raw.header.frame_id = frame_id
        msg_raw.height, msg_raw.width = h, w
        msg_raw.encoding, msg_raw.step = 'bgr8', w*3
        msg_raw.data = frame.tobytes()
        self.pub_raw.publish(msg_raw)

        # Narrowed ROI for precise lookahead vectoring
        roi_top = int(h * 0.65)
        roi_bot = int(h * 0.75)
        roi = frame[roi_top:roi_bot, :]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        mask_blue = cv2.inRange(hsv_roi, np.array([95, 80, 50]), np.array([140, 255, 255]))
        lx = self._get_largest_contour_center(mask_blue)

        if lx is None:
            self._stop_motors()
            cv2.putText(frame, "FAILSAFE: LOST BLUE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            self._publish_debug(frame, h, w, frame_id)
            if self.writer: self.writer.write(frame)
            return

        if self.smoothed_lx is None:
            self.smoothed_lx = lx
        self.smoothed_lx = (0.75 * self.smoothed_lx) + (0.25 * lx)
        
        target_x = int(self.get_parameter('line_target_x').value)
        error = self.smoothed_lx - target_x
        
        dt = now - self.last_error_time
        deriv = (error - self.prev_error) / dt if dt > 0 else 0
        
        kp = float(self.get_parameter('kp').value)
        kd = float(self.get_parameter('kd').value)
        base = int(self.get_parameter('base_pwm').value)
        
        turn = (kp * error) + (kd * deriv)
        
        lm = max(25, min(80, int(base + turn)))
        rm = max(25, min(80, int(base - turn)))

        if now - self.last_serial_time >= self.get_parameter('serial_delay').value:
            self._send_serial_cmd(lm, rm)
            self.last_serial_time = now

        self.prev_error = error
        self.last_error_time = now

        cv2.rectangle(frame, (0, roi_top), (w-1, roi_bot), (255, 255, 255), 1)
        y_vis = int((roi_top + roi_bot) / 2)
        
        cv2.line(frame, (lx, roi_bot), (lx, roi_top), (255, 0, 0), 4)
        cv2.circle(frame, (lx, y_vis), 8, (255, 0, 0), -1)
        
        cv2.line(frame, (target_x, h), (target_x, 0), (0, 255, 0), 2)
        cv2.circle(frame, (target_x, y_vis), 10, (0, 255, 0), -1)

        self._publish_debug(frame, h, w, frame_id)
        if self.writer: self.writer.write(frame)
        
        msg_heading = Float32()
        msg_heading.data = float(error)
        self.pub_heading.publish(msg_heading)

    def _publish_debug(self, frame, h, w, frame_id):
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.height, msg.width = h, w
        msg.encoding, msg.step = 'bgr8', w*3
        msg.data = frame.tobytes()
        self.pub_debug.publish(msg)

    def destroy_node(self):
        self._stop_motors()
        if self.ser: self.ser.close()
        if self.writer: self.writer.release()
        self.cap.release()
        super().destroy_node()

def main():
    rclpy.init()
    node = LineDetectorNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__': main()