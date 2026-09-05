#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix, NavSatStatus
import math


class OdomToNavSat(Node):

    def __init__(self):
        super().__init__('odom_to_navsat')

        # 🔹 Base GPS coordinates (Gwalior approx)
        self.base_lat = 26.2495
        self.base_lon = 78.1741

        # Publisher
        self.pub = self.create_publisher(NavSatFix, '/navsat', 10)

        # Subscriber
        self.sub = self.create_subscription(
            Odometry,
            '/bumperbot_controller/odom',
            self.odom_callback,
            10
        )

        self.get_logger().info("Odom → NavSat node started")

    def odom_callback(self, msg):

        # Current odometry position (meters)
        x = msg.pose.pose.position.x   # East
        y = msg.pose.pose.position.y   # North

        # --- Conversion ---
        delta_lat = y / 111111.0

        delta_lon = x / (111111.0 * math.cos(math.radians(self.base_lat)))

        # Final coordinates
        lat = self.base_lat + delta_lat
        lon = self.base_lon + delta_lon

        # --- Create NavSatFix message ---
        navsat = NavSatFix()

        navsat.header.stamp = self.get_clock().now().to_msg()
        navsat.header.frame_id = "gps_link"

        navsat.status.status = NavSatStatus.STATUS_FIX
        navsat.status.service = NavSatStatus.SERVICE_GPS

        navsat.latitude = lat
        navsat.longitude = lon
        navsat.altitude = 0.0

        # Fake covariance
        navsat.position_covariance = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        ]
        navsat.position_covariance_type = NavSatFix.COVARIANCE_TYPE_APPROXIMATED

        # Publish
        self.pub.publish(navsat)

        self.get_logger().info(
            f"Odom(x={x:.2f}, y={y:.2f}) → Lat={lat:.6f}, Lon={lon:.6f}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = OdomToNavSat()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()