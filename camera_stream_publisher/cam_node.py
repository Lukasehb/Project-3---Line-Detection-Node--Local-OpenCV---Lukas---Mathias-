#!/usr/bin/env python3
import time
import cv2
import numpy as np
import rclpy
import serial
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

class MinimalV4L2Cam(Node):
    def __init__(self):
        super().__init__('rpi_cam_min')

        self.declare_parameter('device', '/dev/video0')
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 15.0)
        self.declare_parameter('fourcc', 'MJPG') 
        self.declare_parameter('frame_id', 'camera_frame')
        self.declare_parameter('topic', '/camera/image_raw')
        
        self.declare_parameter('port', '/dev/ttyACM0')
        self.declare_parameter('baud', 57600)
        self.declare_parameter('kp', 0.15)
        self.declare_parameter('base_pwm', 50)
        self.declare_parameter('serial_delay', 0.1)

        dev      = str(self.get_parameter('device').value)
        width    = int(self.get_parameter('width').value)
        height   = int(self.get_parameter('height').value)
        fps      = float(self.get_parameter('fps').value)
        fourcc_s = str(self.get_parameter('fourcc').value)[:4] 
        self.frame_id = str(self.get_parameter('frame_id').value)
        topic    = str(self.get_parameter('topic').value)
        
        port     = str(self.get_parameter('port').value)
        baud     = int(self.get_parameter('baud').value)
        self.kp  = float(self.get_parameter('kp').value)
        self.base_pwm = int(self.get_parameter('base_pwm').value)
        self.serial_delay = float(self.get_parameter('serial_delay').value)

        self.last_serial_time = time.time()

        self.ser = None
        try:
            self.ser = serial.Serial(port, baudrate=baud, timeout=0.1)
            time.sleep(2.0)
            self.get_logger().info(f"Serial connected to {port} @ {baud}")
        except Exception as e:
            self.get_logger().error(f"Failed to open serial {port}: {e}")

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.pub = self.create_publisher(Image, topic, qos)

        self.cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(dev)

        if not self.cap.isOpened():
            raise RuntimeError("VideoCapture open failed")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS,         fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
        if len(fourcc_s) == 4:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc_s))

        self.period = max(1.0 / max(fps, 0.1), 0.001)
        self._last  = time.time()

        self.timer = self.create_timer(self.period, self._tick)

    def _fit_curve(self, lane_lines):
        if not lane_lines:
            return None
        x_coords = []
        y_coords = []
        for x1, y1, x2, y2 in lane_lines:
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
            
        if len(set(y_coords)) < 3:
            return None
            
        poly = np.polyfit(y_coords, x_coords, 2)
        return poly

    def _draw_curve(self, frame, poly, height, color):
        if poly is None:
            return
        ploty = np.linspace(int(height * 0.3), height - 1, int(height * 0.7))
        plotx = poly[0]*ploty**2 + poly[1]*ploty + poly[2]
        
        valid_idx = (plotx >= 0) & (plotx < frame.shape[1])
        plotx = plotx[valid_idx]
        ploty = ploty[valid_idx]
        
        if len(plotx) == 0:
            return
            
        pts = np.array([np.transpose(np.vstack([plotx, ploty]))], np.int32)
        cv2.polylines(frame, pts, False, color, 3)

    def _send_serial_cmd(self, left_pwm, right_pwm):
        if not self.ser or not self.ser.is_open:
            return
        left_pwm = max(-255, min(255, int(left_pwm)))
        right_pwm = max(-255, min(255, int(right_pwm)))
        cmd = f"D {left_pwm} {right_pwm} 1\n"
        try:
            self.ser.write(cmd.encode())
        except Exception as e:
            self.get_logger().warn(f"Serial write error: {e}")

    def _tick(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return

        height, width = frame.shape[:2]

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red_1 = np.array([0, 40, 40])
        upper_red_1 = np.array([15, 255, 255])
        lower_red_2 = np.array([150, 40, 40])
        upper_red_2 = np.array([180, 255, 255])
        
        mask_red_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
        mask_red_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
        red_mask = cv2.bitwise_or(mask_red_1, mask_red_2)

        left_roi = np.zeros_like(red_mask)
        left_polygon = np.array([[
            (0, height),
            (int(width * 0.6), height),
            (int(width * 0.6), int(height * 0.3)),
            (0, int(height * 0.3))
        ]], np.int32)
        cv2.fillPoly(left_roi, left_polygon, 255)
        red_mask = cv2.bitwise_and(red_mask, left_roi)

        red_edges = cv2.Canny(red_mask, 50, 150)
        red_lines_raw = cv2.HoughLinesP(red_edges, 1, np.pi/180, 20, minLineLength=20, maxLineGap=100)

        left_lines = []
        if red_lines_raw is not None:
            for line in red_lines_raw:
                x1, y1, x2, y2 = line[0]
                if x1 != x2:
                    slope = (y2 - y1) / (x2 - x1)
                    if abs(slope) > 0.05: 
                        left_lines.append((x1, y1, x2, y2))
                        
        left_poly = self._fit_curve(left_lines)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        right_roi = np.zeros_like(edges)
        right_polygon = np.array([[
            (int(width * 0.4), height),
            (width, height),
            (width, int(height * 0.3)),
            (int(width * 0.4), int(height * 0.3))
        ]], np.int32)
        cv2.fillPoly(right_roi, right_polygon, 255)
        grey_edges = cv2.bitwise_and(edges, right_roi)

        grey_lines_raw = cv2.HoughLinesP(grey_edges, 1, np.pi/180, 30, minLineLength=30, maxLineGap=150)

        right_lines = []
        if grey_lines_raw is not None:
            for line in grey_lines_raw:
                x1, y1, x2, y2 = line[0]
                if x1 != x2:
                    slope = (y2 - y1) / (x2 - x1)
                    if abs(slope) > 0.05: 
                        right_lines.append((x1, y1, x2, y2))

        right_poly = self._fit_curve(right_lines)

        self._draw_curve(frame, left_poly, height, (0, 0, 255))
        self._draw_curve(frame, right_poly, height, (255, 0, 0))

        y_eval = int(height * 0.35)
        left_x = None
        right_x = None

        if left_poly is not None:
            left_x = int(left_poly[0]*y_eval**2 + left_poly[1]*y_eval + left_poly[2])
            cv2.circle(frame, (left_x, y_eval), 5, (0, 0, 255), -1)
        if right_poly is not None:
            right_x = int(right_poly[0]*y_eval**2 + right_poly[1]*y_eval + right_poly[2])
            cv2.circle(frame, (right_x, y_eval), 5, (255, 0, 0), -1)

        track_width_est = int(width * 0.5)
        target_x = int(width / 2)
        valid_center = False

        if left_x is not None and right_x is not None:
            center_x = int((left_x + right_x) / 2)
            valid_center = True
        elif left_x is not None:
            center_x = left_x + int(track_width_est / 2)
            valid_center = True
        elif right_x is not None:
            center_x = right_x - int(track_width_est / 2)
            valid_center = True
        else:
            cv2.putText(frame, "FAILSAFE: STOP", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        left_motor = 0
        right_motor = 0

        if valid_center:
            cv2.circle(frame, (center_x, y_eval), 8, (0, 255, 0), -1)
            cv2.line(frame, (target_x, height), (center_x, y_eval), (0, 255, 255), 3)
            
            error = center_x - target_x
            turn = error * self.kp
            
            left_motor = self.base_pwm + turn
            right_motor = self.base_pwm - turn

        current_time = time.time()
        if current_time - self.last_serial_time >= self.serial_delay:
            self._send_serial_cmd(left_motor, right_motor)
            self.last_serial_time = current_time

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.height = frame.shape[0]
        msg.width  = frame.shape[1]
        msg.encoding = 'bgr8' 
        msg.is_bigendian = 0
        msg.step = msg.width * 3
        msg.data = frame.tobytes()

        self.pub.publish(msg)

        dt = time.time() - self._last
        sleep_left = self.period - dt
        if sleep_left > 0:
            time.sleep(sleep_left)
        self._last = time.time()

    def destroy_node(self):
        if self.ser and self.ser.is_open:
            self._send_serial_cmd(0, 0)
            self.ser.close()
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        super().destroy_node()

def main():
    rclpy.init()
    node = None
    try:
        node = MinimalV4L2Cam()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node:
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
