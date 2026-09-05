#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class CameraViewer(Node):
    def __init__(self):
        super().__init__('camera_viewer')

        self.bridge = CvBridge()

        self.sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10
        )

        self.get_logger().info('Camera viewer started (OpenCV GUI mode)')

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # 👉 THIS OPENS THE WINDOW
            cv2.imshow('cam_1 color feed', frame)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(str(e))


def main(args=None):
    rclpy.init(args=args)
    node = CameraViewer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()