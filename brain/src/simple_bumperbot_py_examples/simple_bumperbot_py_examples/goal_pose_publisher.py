#!/usr/bin/env python3
"""
Goal Pose Publisher Node

Publishes a goal pose on /goal_pose (geometry_msgs/PoseStamped),
exactly like clicking "2D Goal Pose" in RViz2.

Usage:
  ros2 run simple_bumperbot_py_examples goal_pose_publisher --ros-args \
      -p x:=1.0 -p y:=2.0 -p yaw:=1.57

Parameters:
  x   (float)  – target X in metres  (default 0.0)
  y   (float)  – target Y in metres  (default 0.0)
  z   (float)  – target Z in metres  (default 0.0)
  yaw (float)  – target heading in radians (default 0.0)
  frame_id (str) – reference frame   (default "map")
"""

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


class GoalPosePublisher(Node):
    def __init__(self):
        super().__init__('goal_pose_publisher')

        # Declare parameters with defaults
        self.declare_parameter('x', 9.4)
        self.declare_parameter('y', -2.4)
        self.declare_parameter('z', 0.0)
        self.declare_parameter('yaw', 0.0)
        self.declare_parameter('frame_id', 'map')

        # Create a latched-style publisher (QoS depth 1, transient local)
        self.pub = self.create_publisher(PoseStamped, '/goal_pose', 10)

        # Small delay so subscribers can discover us, then publish
        self.timer = self.create_timer(1.0, self.publish_goal)

    def publish_goal(self):
        # Read parameters
        x = self.get_parameter('x').get_parameter_value().double_value
        y = self.get_parameter('y').get_parameter_value().double_value
        z = self.get_parameter('z').get_parameter_value().double_value
        yaw = self.get_parameter('yaw').get_parameter_value().double_value
        frame_id = self.get_parameter('frame_id').get_parameter_value().string_value

        # Build the PoseStamped message
        goal = PoseStamped()
        goal.header.frame_id = frame_id
        goal.header.stamp = self.get_clock().now().to_msg()

        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = z

        # Convert yaw → quaternion  (roll=0, pitch=0)
        goal.pose.orientation.x = 0.0
        goal.pose.orientation.y = 0.0
        goal.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.orientation.w = math.cos(yaw / 2.0)

        self.pub.publish(goal)
        self.get_logger().info(
            f'Published goal → x={x:.2f}, y={y:.2f}, z={z:.2f}, '
            f'yaw={yaw:.2f} rad ({math.degrees(yaw):.1f}°) '
            f'[frame: {frame_id}]'
        )

        # Cancel the timer – we only need to publish once
        self.timer.cancel()

        # Shut down after a short delay to let the message propagate
        self.create_timer(0.5, lambda: rclpy.shutdown())


def main(args=None):
    rclpy.init(args=args)
    node = GoalPosePublisher()
    rclpy.spin(node)
    node.destroy_node()


if __name__ == '__main__':
    main()
