#!/usr/bin/env python3
"""
Lane-following robot — Single Blue Centre Line.

Detects one blue line painted on the road and outputs manual steering directions.
Automatic serial driving has been disabled to allow independent teleop control.

Detection method:
  - HSV filter with tighter saturation (S>=180) to reject carpet reflections
  - Largest contour centroid replaces raw mask centroid (ignores false positives)
  - Error = centroid_x - frame_centre -> Directional Logging
"""
import os
import time
import cv2
import numpy as np
import rclpy
# import serial # Disabled for manual teleop
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32


class MinimalV4L2Cam(Node):
    def __init__(self):
        super().__init__('rpi_cam_min')

        # ── ROS parameters ────────────────────────────────────────────────
        self.declare_parameter('device',       '/dev/video0')
        self.declare_parameter('width',        640)
        self.declare_parameter('height',       480)
        self.declare_parameter('fps',          15.0)
        self.declare_parameter('fourcc',       'MJPG')
        self.declare_parameter('frame_id',     'camera_frame')
        self.declare_parameter('topic',        '/camera/image_raw')

        self.declare_parameter('port',         '/dev/ttyACM0')
        self.declare_parameter('baud',         57600)
        self.declare_parameter('kp',           0.15)
        self.declare_parameter('kd',           0.03)
        self.declare_parameter('base_pwm',     50)
        self.declare_parameter('deadband',     20)
        self.declare_parameter('serial_delay', 0.05)

        self.declare_parameter('record',       False)
        self.declare_parameter('record_path',  '~/robot_debug.avi')

        dev      = str(self.get_parameter('device').value)
        width    = int(self.get_parameter('width').value)
        height   = int(self.get_parameter('height').value)
        fps      = float(self.get_parameter('fps').value)
        fourcc_s = str(self.get_parameter('fourcc').value)[:4]
        self.frame_id     = str(self.get_parameter('frame_id').value)
        topic             = str(self.get_parameter('topic').value)

        port              = str(self.get_parameter('port').value)
        baud              = int(self.get_parameter('baud').value)
        self.kp           = float(self.get_parameter('kp').value)
        self.kd           = float(self.get_parameter('kd').value)
        self.base_pwm     = int(self.get_parameter('base_pwm').value)
        self.deadband     = int(self.get_parameter('deadband').value)
        self.serial_delay = float(self.get_parameter('serial_delay').value)

        record      = bool(self.get_parameter('record').value)
        record_path = str(self.get_parameter('record_path').value)

        # ── state ─────────────────────────────────────────────────────────
        self.prev_error       = 0
        self.last_error_time  = time.time()
        self.last_serial_time = time.time()
        self.prev_center_x    = None

        # ── serial (DISABLED FOR MANUAL TELEOP) ───────────────────────────
        self.ser = None
        # try:
        #     self.ser = serial.Serial(port, baudrate=baud, timeout=0.1)
        #     time.sleep(2.0)
        #     self.get_logger().info(f"Serial open on {port}")
        # except Exception as e:
        #     self.get_logger().warn(f"Serial failed: {e}")

        # ── camera ────────────────────────────────────────────────────────
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.pub         = self.create_publisher(Image,   topic, qos)
        self.pub_debug   = self.create_publisher(Image,   '/line_detector/debug_image', qos)
        self.pub_heading = self.create_publisher(Float32, '/line_heading', 10)

        self.cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(dev)
        if not self.cap.isOpened():
            raise RuntimeError("VideoCapture open failed")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS,          fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
        if len(fourcc_s) == 4:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc_s))

        self.writer = None
        if record:
            record_path = os.path.expanduser(record_path)
            self.writer = cv2.VideoWriter(
                record_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
            if self.writer.isOpened():
                self.get_logger().info(f"[RECORD] Saving to: {record_path}")
            else:
                self.get_logger().error(f"[RECORD] Could not open: {record_path}")
                self.writer = None

        self.period = max(1.0 / max(fps, 0.1), 0.001)
        self._last  = time.time()
        self.timer  = self.create_timer(self.period, self._tick)

    # ── helpers ───────────────────────────────────────────────────────────

    def _send_serial_cmd(self, left_pwm, right_pwm):
        # DISABLED FOR MANUAL TELEOP
        pass
        # if not self.ser or not self.ser.is_open:
        #     return
        # left_pwm  = max(-255, min(255, int(left_pwm)))
        # right_pwm = max(-255, min(255, int(right_pwm)))
        # try:
        #     self.ser.write(f"D {left_pwm} {right_pwm} 1\n".encode())
        #     self.ser.flush()
        # except Exception:
        #     pass

    def _stop_motors(self):
        """Send stop 5 times to guarantee the Arduino receives it."""
        # DISABLED FOR MANUAL TELEOP
        pass
        # for _ in range(5):
        #     self._send_serial_cmd(0, 0)
        #     time.sleep(0.05)

    # ── main tick ─────────────────────────────────────────────────────────

    def _tick(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return

        now  = time.time()
        h, w = frame.shape[:2]

        # ── 1. BLUE LINE DETECTION ────────────────────────────────────────
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_blue = cv2.inRange(
            hsv, np.array([100, 180, 120]), np.array([130, 255, 255]))

        # ROI: rows 35-78% — full width so the line is tracked in turns too.
        roi_top = int(h * 0.35)
        roi_bot = int(h * 0.78)
        roi = np.zeros_like(mask_blue)
        roi[roi_top:roi_bot, :] = 255
        mask_blue = cv2.bitwise_and(mask_blue, roi)

        # ── 2. LINE POSITION via largest contour centroid ─────────────────
        blue_px = cv2.countNonZero(mask_blue)
        line_x  = None
        if blue_px > 50:
            contours, _ = cv2.findContours(
                mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) > 200:   # ignore tiny blobs
                    M = cv2.moments(largest)
                    if M['m00'] > 0:
                        line_x = int(M['m10'] / M['m00'])

        # ── 3. SMOOTHING ──────────────────────────────────────────────────
        if line_x is not None:
            if self.prev_center_x is None:
                self.prev_center_x = float(line_x)
            smooth_x = int(0.60 * self.prev_center_x + 0.40 * line_x)
            self.prev_center_x = smooth_x
            valid = True
        else:
            smooth_x = w // 2
            valid    = False

        # ── 4. DEBUG OVERLAY ──────────────────────────────────────────────
        frame[mask_blue > 0] = [0, 220, 0]
        cv2.line(frame, (0, roi_top), (w, roi_top), (200, 200, 0), 1)
        cv2.line(frame, (0, roi_bot), (w, roi_bot), (200, 200, 0), 1)
        cv2.line(frame, (w//2, roi_top), (w//2, roi_bot), (100, 100, 100), 1)
        
        if valid:
            cv2.line(frame, (smooth_x, roi_top), (smooth_x, roi_bot),
                     (0, 0, 255), 3)
            cv2.circle(frame, (smooth_x, (roi_top + roi_bot)//2), 8,
                       (0, 255, 0), -1)
        else:
            cv2.putText(frame, "FAILSAFE: STOP", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        error = smooth_x - w // 2
        cv2.putText(frame,
                    f"px:{blue_px}  x:{line_x}  err:{error:+d}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

# ── 5. MANUAL DRIVING LOGIC ───────────────────────────────────────
        if valid:
            if error > self.deadband:
                self.get_logger().info("DIRECTION: TURN RIGHT")
            elif error < -self.deadband:
                self.get_logger().info("DIRECTION: TURN LEFT")
            else:
                self.get_logger().info("DIRECTION: STRAIGHT")

            self.prev_error      = error
            self.last_error_time = now
        else:
            self.get_logger().warn("DIRECTION: LINE LOST - STOP")

        # ── 6. SERIAL (AUTOMATIC DRIVING DISABLED) ────────────────────────
        # if now - self.last_serial_time >= self.serial_delay:
        #     self._send_serial_cmd(lm, rm)
        #     self.last_serial_time = now

        # ── 7. ROS PUBLISH ────────────────────────────────────────────────
        heading_msg = Float32()
        heading_msg.data = float(error) / (w / 2) if valid else 0.0
        self.pub_heading.publish(heading_msg)

        msg = Image()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.height = h;  msg.width = w
        msg.encoding = 'bgr8';  msg.is_bigendian = 0
        msg.step = w * 3;  msg.data = frame.tobytes()
        self.pub.publish(msg)
        self.pub_debug.publish(msg)

        if self.writer is not None:
            self.writer.write(frame)

        # ── 8. FRAME TIMING ───────────────────────────────────────────────
        elapsed = time.time() - self._last
        if elapsed < self.period:
            time.sleep(self.period - elapsed)
        self._last = time.time()

    # ── cleanup ───────────────────────────────────────────────────────────

    def destroy_node(self):
        self.get_logger().info("Shutting down camera node...")
        # self._stop_motors()
        # if self.ser and self.ser.is_open:
        #     self.ser.close()
        if self.writer is not None:
            self.writer.release()
        if self.cap is not None:
            self.cap.release()
        super().destroy_node()


# ── entry point ───────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = MinimalV4L2Cam()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()