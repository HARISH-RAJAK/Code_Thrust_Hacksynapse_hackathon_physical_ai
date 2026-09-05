#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from turtlesim.msg import Pose
import math


class SimpleTurtlesimKinematics(Node):

    def __init__(self):
        super().__init__('simple_turtlesim_kinematics')

        self.turtle1_pose_sub = self.create_subscription(
            Pose,
            '/turtle1/pose',
            self.turtle1PoseCallback,
            10
        )

        self.turtle2_pose_sub = self.create_subscription(
            Pose,
            '/turtle2/pose',
            self.turtle2PoseCallback,
            10
        )

        self.last_turtle1_pose = Pose()
        self.last_turtle2_pose = Pose()

    def turtle1PoseCallback(self, msg):
        self.last_turtle1_pose = msg

    def turtle2PoseCallback(self, msg):
        self.last_turtle2_pose = msg

        # Translation
        Tx = self.last_turtle2_pose.x - self.last_turtle1_pose.x
        Ty = self.last_turtle2_pose.y - self.last_turtle1_pose.y

        # Rotation
        theta_rad = self.last_turtle2_pose.theta - self.last_turtle1_pose.theta
        theta_deg = 180 * theta_rad / 3.14

        # Rotation Matrix
        R11 = math.cos(theta_rad)
        R12 = -math.sin(theta_rad)
        R21 = math.sin(theta_rad)
        R22 = math.cos(theta_rad)

        self.get_logger().info(
            "\n"
            "Translation Vector turtle1 -> turtle2\n"
            "Tx: %f\n"
            "Ty: %f\n\n"
            "Rotation Matrix turtle1 -> turtle2\n"
            "theta(rad): %f\n"
            "theta(deg): %f\n\n"
            "|R11   R12| : |%f   %f|\n"
            "|R21   R22| : |%f   %f|\n"
            % (Tx, Ty, theta_rad, theta_deg, R11, R12, R21, R22)
        )


def main():
    rclpy.init()
    simple_turtlesim_kinematics = SimpleTurtlesimKinematics()
    rclpy.spin(simple_turtlesim_kinematics)
    simple_turtlesim_kinematics.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
