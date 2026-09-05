#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import json
import os
from datetime import datetime

from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster, Buffer, TransformListener, LookupException
import tf2_geometry_msgs.tf2_geometry_msgs
from message_filters import Subscriber, ApproximateTimeSynchronizer
from rclpy.time import Time
from std_srvs.srv import Trigger


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

        # -------- ArUco --------
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # ArUco marker size in meters (adjust based on your marker)
        self.marker_size = 0.05  # 5cm marker

        # -------- Eye-in-Hand Calibration --------
        self.calibration_samples = []  # List of (gripper_pose, marker_pose) tuples
        self.gripper_frame = "gripper_base_1"  # End-effector frame
        self.base_frame = "base_Link"  # Robot base frame
        self.camera_link = "zed_camera_link"  # Camera link attached to gripper
        self.calibration_result = None  # Transformation from gripper_base_1 to zed_camera_link
        self.calibration_file = os.path.expanduser("~/eye_in_hand_calibration.json")
        self.is_calibrated = False
        
        # Load existing calibration if available
        self.load_calibration()
        
        # Services for calibration
        self.capture_srv = self.create_service(
            Trigger, 
            'capture_calibration_sample', 
            self.capture_sample_callback
        )
        self.compute_srv = self.create_service(
            Trigger, 
            'compute_calibration', 
            self.compute_calibration_callback
        )
        self.reset_srv = self.create_service(
            Trigger, 
            'reset_calibration', 
            self.reset_calibration_callback
        )

        self.get_logger().info("Fruit + ArUco TF node started")
        self.get_logger().info("Eye-in-Hand Calibration Services:")
        self.get_logger().info("  - ros2 service call /capture_calibration_sample std_srvs/srv/Trigger")
        self.get_logger().info("  - ros2 service call /compute_calibration std_srvs/srv/Trigger")
        self.get_logger().info("  - ros2 service call /reset_calibration std_srvs/srv/Trigger")
        self.get_logger().info(f"Samples needed: 10-15 from different poses")
        self.get_logger().info(f"Calibrating: {self.gripper_frame} -> {self.camera_link}")

    # --------------------------------------------------
    def caminfo_callback(self, msg: CameraInfo):

        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

        self.get_logger().info("Camera intrinsics received")
        self.destroy_subscription(self.caminfo_sub)

    # --------------------------------------------------
    # Eye-in-Hand Calibration Methods
    # --------------------------------------------------
    
    def get_transform_matrix(self, transform):
        """Convert TransformStamped to 4x4 transformation matrix"""
        t = transform.transform
        # Translation
        trans = np.array([t.translation.x, t.translation.y, t.translation.z])
        # Rotation (quaternion to rotation matrix)
        q = [t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w]
        rot = self.quaternion_to_rotation_matrix(q)
        
        # Build 4x4 matrix
        mat = np.eye(4)
        mat[:3, :3] = rot
        mat[:3, 3] = trans
        return mat
    
    def quaternion_to_rotation_matrix(self, q):
        """Convert quaternion [x, y, z, w] to 3x3 rotation matrix"""
        x, y, z, w = q
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])
    
    def rotation_matrix_to_quaternion(self, R):
        """Convert 3x3 rotation matrix to quaternion [x, y, z, w]"""
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s
        return [x, y, z, w]
    
    def capture_sample_callback(self, request, response):
        """Service callback to capture a calibration sample"""
        try:
            # Get gripper pose (base to gripper)
            gripper_tf = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.gripper_frame,
                Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            
            # Get marker pose (camera_link to marker)
            marker_tf = self.tf_buffer.lookup_transform(
                self.camera_link,
                "aruco_1",
                Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            
            # Convert to matrices
            gripper_mat = self.get_transform_matrix(gripper_tf)
            marker_mat = self.get_transform_matrix(marker_tf)
            
            # Store sample
            self.calibration_samples.append({
                'gripper': gripper_mat.tolist(),
                'marker': marker_mat.tolist(),
                'timestamp': datetime.now().isoformat()
            })
            
            response.success = True
            response.message = f"Sample {len(self.calibration_samples)} captured! Need 10-15 samples from different poses."
            self.get_logger().info(response.message)
            
        except LookupException as e:
            response.success = False
            response.message = f"Failed to get transforms: {str(e)}"
            self.get_logger().error(response.message)
        except Exception as e:
            response.success = False
            response.message = f"Error capturing sample: {str(e)}"
            self.get_logger().error(response.message)
            
        return response
    
    def compute_calibration_callback(self, request, response):
        """Service callback to compute hand-eye calibration"""
        if len(self.calibration_samples) < 3:
            response.success = False
            response.message = f"Need at least 3 samples, have {len(self.calibration_samples)}"
            self.get_logger().warn(response.message)
            return response
        
        try:
            # Prepare data for calibrateHandEye
            R_gripper2base = []
            t_gripper2base = []
            R_marker2cam = []
            t_marker2cam = []
            
            for sample in self.calibration_samples:
                gripper_mat = np.array(sample['gripper'])
                marker_mat = np.array(sample['marker'])
                
                # Extract rotation and translation
                R_gripper2base.append(gripper_mat[:3, :3])
                t_gripper2base.append(gripper_mat[:3, 3].reshape(3, 1))
                R_marker2cam.append(marker_mat[:3, :3])
                t_marker2cam.append(marker_mat[:3, 3].reshape(3, 1))
            
            # Compute hand-eye calibration (eye-in-hand)
            R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                R_gripper2base=R_gripper2base,
                t_gripper2base=t_gripper2base,
                R_target2cam=R_marker2cam,
                t_target2cam=t_marker2cam,
                method=cv2.CALIB_HAND_EYE_TSAI
            )
            
            # Build transformation matrix
            self.calibration_result = np.eye(4)
            self.calibration_result[:3, :3] = R_cam2gripper
            self.calibration_result[:3, 3] = t_cam2gripper.flatten()
            
            # Save calibration
            self.save_calibration()
            self.is_calibrated = True
            
            response.success = True
            response.message = f"Calibration computed using {len(self.calibration_samples)} samples and saved to {self.calibration_file}"
            self.get_logger().info(response.message)
            self.get_logger().info(f"{self.gripper_frame} to {self.camera_link} Transform:\n{self.calibration_result}")
            
        except Exception as e:
            response.success = False
            response.message = f"Failed to compute calibration: {str(e)}"
            self.get_logger().error(response.message)
            
        return response
    
    def reset_calibration_callback(self, request, response):
        """Service callback to reset calibration samples"""
        self.calibration_samples = []
        response.success = True
        response.message = "Calibration samples reset"
        self.get_logger().info(response.message)
        return response
    
    def save_calibration(self):
        """Save calibration to file"""
        if self.calibration_result is None:
            return
        
        data = {
            'calibration_matrix': self.calibration_result.tolist(),
            'num_samples': len(self.calibration_samples),
            'timestamp': datetime.now().isoformat(),
            'gripper_frame': self.gripper_frame,
            'base_frame': self.base_frame,
            'camera_link': self.camera_link,
            'description': f'Transform from {self.gripper_frame} to {self.camera_link}'
        }
        
        with open(self.calibration_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.get_logger().info(f"Calibration saved to {self.calibration_file}")
    
    def load_calibration(self):
        """Load calibration from file"""
        if not os.path.exists(self.calibration_file):
            self.get_logger().info("No existing calibration file found")
            return
        
        try:
            with open(self.calibration_file, 'r') as f:
                data = json.load(f)
            
            self.calibration_result = np.array(data['calibration_matrix'])
            self.is_calibrated = True
            self.get_logger().info(f"Calibration loaded from {self.calibration_file}")
            self.get_logger().info(f"Calibration from {data['timestamp']} using {data['num_samples']} samples")
            
        except Exception as e:
            self.get_logger().error(f"Failed to load calibration: {str(e)}")

    # --------------------------------------------------
    def synced_callback(self, img_msg: Image, depth_msg: Image):

        if self.fx is None:
            return

        frame = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, '32FC1')

        # =================================================
        # 🔵 ARUCO DETECTION WITH ORIENTATION
        # =================================================
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is not None:
            # Prepare camera matrix and distortion coefficients
            camera_matrix = np.array([
                [self.fx, 0, self.cx],
                [0, self.fy, self.cy],
                [0, 0, 1]
            ], dtype=np.float32)
            dist_coeffs = np.zeros((5,), dtype=np.float32)  # Assuming no distortion

            for i in range(len(ids)):

                if ids[i][0] != 0:
                    continue

                marker_corners = corners[i][0]
                
                # Define 3D points of the marker corners in marker coordinate system
                half_size = self.marker_size / 2.0
                obj_points = np.array([
                    [-half_size,  half_size, 0],
                    [ half_size,  half_size, 0],
                    [ half_size, -half_size, 0],
                    [-half_size, -half_size, 0]
                ], dtype=np.float32)
                
                # Estimate pose using solvePnP
                success, rvec, tvec = cv2.solvePnP(
                    obj_points, 
                    marker_corners, 
                    camera_matrix, 
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
                
                # Convert rotation vector to rotation matrix
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                
                # Convert rotation matrix to quaternion
                # Using the formula for rotation matrix to quaternion conversion
                trace = np.trace(rotation_matrix)
                
                if trace > 0:
                    s = 0.5 / np.sqrt(trace + 1.0)
                    qw = 0.25 / s
                    qx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) * s
                    qy = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) * s
                    qz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) * s
                else:
                    if rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
                        s = 2.0 * np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2])
                        qw = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
                        qx = 0.25 * s
                        qy = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
                        qz = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
                    elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
                        s = 2.0 * np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2])
                        qw = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
                        qx = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
                        qy = 0.25 * s
                        qz = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
                    else:
                        s = 2.0 * np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1])
                        qw = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
                        qx = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
                        qy = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
                        qz = 0.25 * s

                # Create and publish TF wrt zed_left_camera_frame
                t = TransformStamped()
                t.header.stamp = img_msg.header.stamp
                t.header.frame_id = "zed_left_camera_frame_optical"
                t.child_frame_id = "aruco_1"

                t.transform.translation.x = float(tvec[0])
                t.transform.translation.y = float(tvec[1])
                t.transform.translation.z = float(tvec[2])
                
                t.transform.rotation.x = float(qx)
                t.transform.rotation.y = float(qy)
                t.transform.rotation.z = float(qz)
                t.transform.rotation.w = float(qw)

                self.br.sendTransform(t)

                # Draw marker for visualization
                cv2.polylines(
                    frame,
                    [marker_corners.astype(np.int32)],
                    True,
                    (255,0,0),
                    2
                )
                
                # Draw axis on marker
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)

        # Display calibration info
        status_color = (0, 255, 0) if self.is_calibrated else (0, 165, 255)
        status_text = f"Calibrated" if self.is_calibrated else f"Not Calibrated"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, status_color, 2)
        cv2.putText(frame, f"Samples: {len(self.calibration_samples)}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Fruit + ArUco", frame)
        cv2.waitKey(1)


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