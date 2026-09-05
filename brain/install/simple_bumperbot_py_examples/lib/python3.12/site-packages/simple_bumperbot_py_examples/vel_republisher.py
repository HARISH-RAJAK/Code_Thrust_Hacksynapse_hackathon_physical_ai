#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, TwistStamped


class CmdVelToStamped(Node):

    def __init__(self):
        super().__init__('cmd_vel_to_stamped')

        self.subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_callback,
            10
        )

        self.publisher = self.create_publisher(
            TwistStamped,
            '/bumperbot_controller/cmd_vel',
            10
        )

        self.get_logger().info('cmd_vel → TwistStamped converter started')

    def cmd_callback(self, msg):
        stamped_msg = TwistStamped()

        stamped_msg.header.stamp = self.get_clock().now().to_msg()
        stamped_msg.header.frame_id = 'base_link'

        stamped_msg.twist = msg

        self.publisher.publish(stamped_msg)


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelToStamped()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()