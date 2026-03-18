# Line Detection Node — Local OpenCV

**Authors:** Lukas & Mathias

## Description
A ROS2 node running on a Raspberry Pi that drives a robot autonomously around a track using a camera and OpenCV.

## How it works
- Detects the **left and right white border walls** using HSV colour masking
- Detects **red turn markers** that replace the white border at corners
- Uses a **PD controller** to steer the robot toward the centre of the lane
- Publishes the annotated camera image over ROS2 for debugging

## Requirements
- ROS2 (Humble or later)
- Python 3
- OpenCV (`cv2`)
- PySerial

## Launch
```bash
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash

ros2 run camera_stream_publisher cam --ros-args \
  -p device:=/dev/video0 \
  -p width:=640 \
  -p height:=480 \
  -p fps:=15.0 \
  -p fourcc:=MJPG \
  -p record:=true
```

## Parameters
| Parameter | Default | Description |
|---|---|---|
| `kp` | 0.15 | Proportional gain |
| `kd` | 0.03 | Derivative gain |
| `base_pwm` | 50 | Base motor speed |
| `deadband` | 20 | Error deadband (px) |
| `record` | false | Save debug video |
| `record_path` | ~/robot_debug.avi | Video save path |
