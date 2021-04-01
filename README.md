<p align="center">
  <h1 align="center">ROS 2 Deep Learning Package for Robot Movement</h1>
</p>

ROS 2 Package to Publish Twist Message from Subscribed Sensor Image Message using End to End Learning (Deep Learning) Method for Robot Movement.

## Colaborators
[Computer Fusion Laboratory (CFL) - Temple University College of Engineering](https://sites.temple.edu/cflab/people/)
* [Animesh Bala Ani](https://animeshani.com/)
* [Dr. Li Bai](https://engineering.temple.edu/about/faculty-staff/li-bai-lbai)

## Dependency
Deep Learning Python Modules.</br>
Can be installed globally followng https://github.com/ANI717/race-car#env-rasp in Raspberry Pi 4.</br>
```
PyTorch
OpenCV
NumPy
Pandas
Matplotlib
Json
```
Deep Learning Tools (Ref: https://github.com/ANI717/race-car).
```
"deeplearn" directory
```
ROS 2 Modules (Comes with ROS 2 installation)
```
rclpy
sensor_msgs
geometry_msgs
ament_index_python.packages
```
ROS 2 Package to capture image with camera
```
ros2 run image_tools cam2image
```
