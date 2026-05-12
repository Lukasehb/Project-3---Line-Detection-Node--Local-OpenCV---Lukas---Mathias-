# Line Detection Node (Local OpenCV)

Project 3: ROS 2 Autonomous Line Follower

Authors: Lukas & Matthias

## 1. Project Overview

This repository contains a ROS 2 package featuring a Line Detection Node. The robot uses a camera to detect a single blue centre line on the track and calculates heading corrections in real-time to maintain its path. The system includes a closed-loop controller and a visual failsafe mechanism.

## 2. Technical Implementation

### Detection Pipeline
The `cam_node.py` implements a robust OpenCV pipeline:

- Pre-processing: Converts the raw BGR image to HSV color space to isolate the blue track markers.
- ROI (Region of Interest): To ensure stability and focus, the node only processes the bottom 20% of the frame (where the track is closest to the robot).
- Noise Reduction: Applies morphological "Opening" (erosion followed by dilation) to remove small artifacts and reflections.
- Bottom-Slice Lock: The node isolates the lowest 30 pixels of the ROI and calculates the Center of Mass (Moments) of the blue contour. This "look-ahead" region ensures a stable heading calculation even during sharp turns.

### Controller & Smoothing

- Smoothing Algorithm: A weighted average (60% new / 40% old) is applied to the detected X-position to reduce jitter while remaining responsive to rapid track changes.
- Heading Calculation: Calculates the error relative to the frame center.
- Proportional Control: Publishes `geometry_msgs/Twist` messages to `/cmd_vel`. The angular velocity is proportional to the track error, allowing for smooth steering.
- Deadband: A 20-pixel deadband prevents the motors from "hunting" or vibrating when the robot is perfectly centered.

### Failsafe & Debugging

- Failsafe Engine Cut: If the blue line is lost (less than 50 pixels detected), the node automatically sets linear and angular velocity to `0.0` and triggers a "LINE LOST - STOP" warning.
- Debug Image: Publishes a processed stream to `/line_detector/debug_image` with:
  - Green mask overlay of the detected blue line.
  - Cyan ROI boundaries.
  - A red indicator for the detected center and a green dot for the smoothed target.
  - Real-time telemetry (pixel count, error, and current direction).

## 3. Hardware & Software

- Hardware: Mobile Robot with a V4L2-compatible camera (Raspberry Pi).
- OS: Ubuntu 22.04 with ROS 2 Humble.
- Libraries: Python 3, OpenCV (`cv2`), NumPy.

## 4. Interfaces

| Type | Topic | Message Type | Description |
|---|---|---|---|
| Input | `/camera/image_raw` | `sensor_msgs/Image` | Raw camera stream |
| Output | `/line_heading` | `std_msgs/Float32` | Normalized heading error |
| Output | `/cmd_vel` | `geometry_msgs/Twist` | Velocity commands for the robot (optional) |
| Debug | `/line_detector/debug_image` | `sensor_msgs/Image` | Annotated visual feedback |

## 5. Installation & Usage

### Build

```bash
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

### Run

Launch the camera node with default or custom parameters:

```bash
cd ~/Project-3---Line-Detection-Node--Local-OpenCV---Lukas---Mathias-
colcon build --symlink-install
source install/setup.bash

# with all parameters + recording
ros2 run camera_stream_publisher cam_node --ros-args -p device:=/dev/video0 -p width:=640 -p height:=480 -p fps:=15.0 -p fourcc:=MJPG -p record:=false



cd ~/Project-3---Line-Detection-Node--Local-OpenCV---Lukas---Mathias-
colcon build
source install/setup.bash
ros2 run my_robot auto_drive
```

## 6. rqt with ssh

Reconnect with the -X or -Y flag:

Bash
ssh -X ubuntu@<rpi_ip_address>
Verify the DISPLAY variable:
Once logged in, run:

Bash
    echo $DISPLAY
    ```
    It should return something like `localhost:10.0`. If it is empty, X11 forwarding is not active.

### Persistent Fix: Install X11 Backend
If running directly on the Pi with a monitor and the error persists, ensure the X11 development libraries are installed:

```bash
sudo apt update
sudo apt install libxcb-xinerama0
```
Headless Operation (No Monitor)
To run rqt without a physical display, use a virtual framebuffer like xvfb or the "offscreen" plugin (though offscreen will not show a UI):

Using the Offscreen plugin (to test if the app runs):

 ```Bash
export QT_QPA_PLATFORM=offscreen
rqt
```
Using VNC:
Install and start a VNC server (like tightvncserver or wayvnc), connect via a VNC client, and run the command within that desktop session.

Environment Variable Override
Force the application to use the X11 plugin specifically if multiple backends are present:

```Bash
export QT_QPA_PLATFORM=xcb
rqt
```

## 7. Parameters

| Parameter | Default | Description |
|---|---:|---|
| `kp` | `0.05` | Proportional gain for steering |
| `base_pwm` | `10` | Base motor power (unused in automatic control example) |
| `deadband` | `20` | Pixel threshold before steering kicks in |
| `record` | `False` | Toggle video recording to file |
| `record_path` | `~/robot_debug.avi` | Storage path for debug video |
