#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np

from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
import tf2_geometry_msgs.tf2_geometry_msgs
from message_filters import Subscriber, ApproximateTimeSynchronizer
import rclpy.duration
from rclpy.time import Time


class CameraViewer(Node):

    def __init__(self):
        super().__init__('fruit_tf_publisher')

        self.bridge = CvBridge()

        # Camera intrinsics
        self.fx = self.fy = self.cx = self.cy = None

        # TF
        self.br = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # -------- Camera Info --------
        self.caminfo_sub = self.create_subscription(
            CameraInfo,
            '/zed/zed_node/rgb/color/rect/camera_info',
            self.caminfo_callback,
            10
        )

        # -------- RGB + Depth Sync --------
        self.rgb_sub = Subscriber(self, Image, '/zed/zed_node/rgb/color/rect/image')
        self.depth_sub = Subscriber(self, Image, '/zed/zed_node/depth/depth_registered')

        self.ts = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.synced_callback)

        self.get_logger().info("Fruit detection + TF node started")

    # --------------------------------------------------
    # Camera Info Callback
    # --------------------------------------------------
    def caminfo_callback(self, msg: CameraInfo):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

        self.get_logger().info("Camera intrinsics received")
        self.destroy_subscription(self.caminfo_sub)

    # --------------------------------------------------
    # RGB + Depth Callback
    # --------------------------------------------------
    def synced_callback(self, img_msg: Image, depth_msg: Image):

        if self.fx is None:
            self.get_logger().warn("Waiting for camera intrinsics...")
            return

        self.get_logger().info("Processing frame...", throttle_duration_sec=2.0)
        
        frame = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, '32FC1')

        # =================================================
        # 🔴 OBJECT DETECTION (Bad Fruit + Gray Tubes)
        # =================================================
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # -------- Detect Bad Fruit --------
        lower_bad_fruit = np.array([0, 114, 44])
        upper_bad_fruit = np.array([15, 255, 191])

        mask_bad_fruit = cv2.inRange(hsv, lower_bad_fruit, upper_bad_fruit)
        contours_bad_fruit, _ = cv2.findContours(mask_bad_fruit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # -------- Detect Gray Tubes --------
        lower_gray = np.array([0, 99, 64])
        upper_gray = np.array([179, 173, 149])

        mask_gray = cv2.inRange(hsv, lower_gray, upper_gray)
        contours_gray, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        fruit_id = 0
        tube_id = 0

        # -------- Process Bad Fruits --------
        # for cnt in contours_bad_fruit:

        #     if cv2.contourArea(cnt) < 500:
        #         continue

        #     x, y, w, h = cv2.boundingRect(cnt)

        #     u = int(x + w / 2)
        #     v = int(y + h / 2)

        #     Z = np.nanmedian(depth[y:y+h, x:x+w])

        #     if np.isnan(Z) or Z <= 0.1:
        #         continue

        #     # 3D projection (camera frame)
        #     X = (u - self.cx) * Z / self.fx
        #     Y = (v - self.cy) * Z / self.fy

        #     # Adjust axis to match your convention
        #     X_link = Z
        #     Y_link = -X
        #     Z_link = -Y

        #     pose_cam = PoseStamped()
        #     # Use image frame, but query TF at latest available time
        #     pose_cam.header.stamp = img_msg.header.stamp
        #     pose_cam.header.frame_id = "zed_left_camera_frame"

        #     pose_cam.pose.position.x = float(X_link)
        #     pose_cam.pose.position.y = float(Y_link)
        #     pose_cam.pose.position.z = float(Z_link)

        #     # Identity orientation in optical frame (no rotation needed)
        #     pose_cam.pose.orientation.x = 0.0
        #     pose_cam.pose.orientation.y = 0.0
        #     pose_cam.pose.orientation.z = 0.0
        #     pose_cam.pose.orientation.w = 1.0

        #     fruit_id += 1
        #     fruit_frame = f"bad_fruit_{fruit_id}"

        #     try:
        #         # Get latest available transform between base_Link and camera frame
        #         tf_cam_to_base = self.tf_buffer.lookup_transform(
        #             "base_Link",
        #             pose_cam.header.frame_id,
        #             Time())  # Time() == latest

        #         pose_base = tf2_geometry_msgs.tf2_geometry_msgs.do_transform_pose_stamped(
        #             pose_cam,
        #             tf_cam_to_base
        #         )

        #         t = TransformStamped()
        #         # Stamp with latest TF time used above
        #         t.header.stamp = tf_cam_to_base.header.stamp
        #         t.header.frame_id = "base_Link"
        #         t.child_frame_id = fruit_frame

        #         t.transform.translation.x = pose_base.pose.position.x
        #         t.transform.translation.y = pose_base.pose.position.y - 0.00  # Adjust for fruit side
        #         t.transform.translation.z = pose_base.pose.position.z - 0.00  # Adjust for fruit height
        #         t.transform.rotation = pose_base.pose.orientation

        #         self.br.sendTransform(t)

        #     except Exception as e:
        #         self.get_logger().warn(f"TF transform failed: {e}")

        #     # Visualization
        #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        #     cv2.putText(frame, "bad fruit", (x, y-10),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # -------- Process Gray Tubes --------
        for cnt in contours_gray:

            if cv2.contourArea(cnt) < 500:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            u = int(x + w / 2)
            v = int(y + h / 2)

            Z = np.nanmedian(depth[y:y+h, x:x+w])

            if np.isnan(Z) or Z <= 0.1:
                continue

            # 3D projection (camera frame)
            X = (u - self.cx) * Z / self.fx
            Y = (v - self.cy) * Z / self.fy

            # Adjust axis to match your convention
            X_link = Z
            Y_link = -X
            Z_link = -Y

            pose_cam = PoseStamped()
            # Use image frame, but query TF at latest available time
            pose_cam.header.stamp = img_msg.header.stamp
            pose_cam.header.frame_id = "zed_left_camera_frame"

            pose_cam.pose.position.x = float(X_link)
            pose_cam.pose.position.y = float(Y_link)
            pose_cam.pose.position.z = float(Z_link)

            # Fixed orientation (as per your previous logic)
            pose_cam.pose.orientation.x = 0.0
            pose_cam.pose.orientation.y = 0.0
            pose_cam.pose.orientation.z = 0.0
            pose_cam.pose.orientation.w = 1.0

            tube_id += 1
            tube_frame = f"tube_{tube_id}"

            try:
                # Get latest available transform between base_Link and camera frame
                tf_cam_to_base = self.tf_buffer.lookup_transform(
                    "base_Link",
                    pose_cam.header.frame_id,
                    Time())  # Time() == latest

                pose_base = tf2_geometry_msgs.tf2_geometry_msgs.do_transform_pose_stamped(
                    pose_cam,
                    tf_cam_to_base
                )

                t = TransformStamped()
                # Stamp with latest TF time used above
                t.header.stamp = tf_cam_to_base.header.stamp
                t.header.frame_id = "base_Link"
                t.child_frame_id = tube_frame

                t.transform.translation.x = pose_base.pose.position.x
                t.transform.translation.y = pose_base.pose.position.y -0.00
                t.transform.translation.z = pose_base.pose.position.z + 0.02  # Adjust for tube height
                t.transform.rotation = pose_base.pose.orientation

                self.br.sendTransform(t)

            except Exception as e:
                self.get_logger().warn(f"TF transform failed: {e}")

            # Visualization
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
            cv2.putText(frame, "tube", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

        # Always show the frame (even if no fruits detected)
        try:
            cv2.namedWindow("Fruit Detection", cv2.WINDOW_NORMAL)
            cv2.imshow("Fruit Detection", frame)
            cv2.waitKey(1)
        except cv2.error as e:
            self.get_logger().error(f"OpenCV window error: {e}")
        
        self.get_logger().info(f"Detected {fruit_id} bad fruits, {tube_id} tubes", throttle_duration_sec=2.0)


# --------------------------------------------------
# Main
# --------------------------------------------------
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