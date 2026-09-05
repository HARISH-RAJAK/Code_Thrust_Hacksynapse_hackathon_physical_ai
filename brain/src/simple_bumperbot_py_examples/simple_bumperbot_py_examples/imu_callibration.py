#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import time
from collections import deque
import math


class IMUCalibrationNode(Node):
    def __init__(self):
        super().__init__('imu_calibration_node')
        
        # Subscriber
        self.subscription = self.create_subscription(
            Imu,
            '/imu/out',
            self.imu_callback,
            10)
        
        # Publisher
        self.publisher = self.create_publisher(
            Imu,
            '/imu/calibrated',
            10)
        
        # Calibration parameters
        self.calibration_duration = 3.0  # seconds
        self.calibration_data = deque()
        self.calibration_start_time = None
        self.is_calibrated = False
        
        # Baseline values (will be set after calibration)
        self.baseline_q_x = 0.0
        self.baseline_q_y = 0.0
        self.baseline_q_z = 0.0
        self.baseline_q_w = 1.0
        
        self.baseline_ax = 0.0
        self.baseline_ay = 0.0
        self.baseline_az = 0.0
        
        self.baseline_wx = 0.0
        self.baseline_wy = 0.0
        self.baseline_wz = 0.0
        
        self.get_logger().info('IMU Calibration Node started. Collecting calibration data for 3 seconds...')
        self.calibration_start_time = time.time()
    
    def imu_callback(self, msg):
        """Callback to handle incoming IMU messages"""
        
        if not self.is_calibrated:
            # Collecting calibration data
            elapsed_time = time.time() - self.calibration_start_time
            
            if elapsed_time < self.calibration_duration:
                # Store data for calibration
                self.calibration_data.append({
                    'q_x': msg.orientation.x,
                    'q_y': msg.orientation.y,
                    'q_z': msg.orientation.z,
                    'q_w': msg.orientation.w,
                    'ax': msg.linear_acceleration.x,
                    'ay': msg.linear_acceleration.y,
                    'az': msg.linear_acceleration.z,
                    'wx': msg.angular_velocity.x,
                    'wy': msg.angular_velocity.y,
                    'wz': msg.angular_velocity.z,
                })
            else:
                # Calibration period complete - calculate average
                self.compute_calibration_baseline()
                self.is_calibrated = True
                self.get_logger().info('Calibration complete! Publishing calibrated IMU data...')
        
        if self.is_calibrated:
            # Publish calibrated data
            calibrated_msg = self.calibrate_imu(msg)
            self.publisher.publish(calibrated_msg)
    
    def compute_calibration_baseline(self):
        """Calculate average values from calibration data"""
        if len(self.calibration_data) == 0:
            self.get_logger().warn('No calibration data collected')
            return
        
        # Convert quaternions to Euler angles for averaging
        euler_angles = []
        for data in self.calibration_data:
            qx, qy, qz, qw = data['q_x'], data['q_y'], data['q_z'], data['q_w']
            roll, pitch, yaw = self.quaternion_to_euler(qx, qy, qz, qw)
            euler_angles.append({
                'roll': roll,
                'pitch': pitch,
                'yaw': yaw,
                'ax': data['ax'],
                'ay': data['ay'],
                'az': data['az'],
                'wx': data['wx'],
                'wy': data['wy'],
                'wz': data['wz'],
            })
        
        # Calculate averages
        avg_roll = sum(e['roll'] for e in euler_angles) / len(euler_angles)
        avg_pitch = sum(e['pitch'] for e in euler_angles) / len(euler_angles)
        avg_yaw = sum(e['yaw'] for e in euler_angles) / len(euler_angles)
        
        # Convert back to quaternion
        self.baseline_q_x, self.baseline_q_y, self.baseline_q_z, self.baseline_q_w = \
            self.euler_to_quaternion(avg_roll, avg_pitch, avg_yaw)
        
        # Average accelerations and angular velocities
        self.baseline_ax = sum(e['ax'] for e in euler_angles) / len(euler_angles)
        self.baseline_ay = sum(e['ay'] for e in euler_angles) / len(euler_angles)
        avg_az = sum(e['az'] for e in euler_angles) / len(euler_angles)
        # Z-axis: store the difference from 9.8 (gravity)
        self.baseline_az = avg_az - 9.8
        
        self.baseline_wx = sum(e['wx'] for e in euler_angles) / len(euler_angles)
        self.baseline_wy = sum(e['wy'] for e in euler_angles) / len(euler_angles)
        self.baseline_wz = sum(e['wz'] for e in euler_angles) / len(euler_angles)
        
        self.get_logger().info(
            f'Baseline calculated from {len(self.calibration_data)} samples:\n'
            f'  Orientation (Euler): roll={avg_roll:.4f}, pitch={avg_pitch:.4f}, yaw={avg_yaw:.4f}\n'
            f'  Baseline Accel: ax={self.baseline_ax:.4f}, ay={self.baseline_ay:.4f}, az_offset={self.baseline_az:.4f}\n'
            f'  Avg sensor z-axis: {avg_az:.4f}, will publish as 9.8 when at rest'
        )
    
    def calibrate_imu(self, msg):
        """Apply calibration to incoming IMU message"""
        calibrated_msg = Imu()
        calibrated_msg.header = msg.header
        calibrated_msg.header.frame_id = msg.header.frame_id
        
        # Convert current and baseline orientations to Euler angles
        qx, qy, qz, qw = msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        current_roll, current_pitch, current_yaw = self.quaternion_to_euler(qx, qy, qz, qw)
        
        # Baseline orientation in Euler (from baseline quaternion)
        baseline_roll, baseline_pitch, baseline_yaw = self.quaternion_to_euler(
            self.baseline_q_x, self.baseline_q_y, self.baseline_q_z, self.baseline_q_w)
        
        # Calculate difference in Euler angles
        delta_roll = current_roll - baseline_roll
        delta_pitch = current_pitch - baseline_pitch
        delta_yaw = current_yaw - baseline_yaw
        
        # Convert difference back to quaternion (starting from identity)
        cal_qx, cal_qy, cal_qz, cal_qw = self.euler_to_quaternion(
            delta_roll, delta_pitch, delta_yaw)
        
        calibrated_msg.orientation.x = cal_qx
        calibrated_msg.orientation.y = cal_qy
        calibrated_msg.orientation.z = cal_qz
        calibrated_msg.orientation.w = cal_qw
        
        # Copy covariance
        calibrated_msg.orientation_covariance = msg.orientation_covariance
        
        # Subtract baseline from angular velocity (difference only)
        calibrated_msg.angular_velocity.x = msg.angular_velocity.x - self.baseline_wx
        calibrated_msg.angular_velocity.y = msg.angular_velocity.y - self.baseline_wy
        calibrated_msg.angular_velocity.z = msg.angular_velocity.z - self.baseline_wz
        calibrated_msg.angular_velocity_covariance = msg.angular_velocity_covariance
        
        # Subtract baseline from linear acceleration (difference only)
        calibrated_msg.linear_acceleration.x = msg.linear_acceleration.x - self.baseline_ax
        calibrated_msg.linear_acceleration.y = msg.linear_acceleration.y - self.baseline_ay
        calibrated_msg.linear_acceleration.z = msg.linear_acceleration.z - self.baseline_az
        calibrated_msg.linear_acceleration_covariance = msg.linear_acceleration_covariance
        
        return calibrated_msg
    
    @staticmethod
    def quaternion_to_euler(qx, qy, qz, qw):
        """Convert quaternion to Euler angles (roll, pitch, yaw)"""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw
    
    @staticmethod
    def euler_to_quaternion(roll, pitch, yaw):
        """Convert Euler angles (roll, pitch, yaw) to quaternion"""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        return qx, qy, qz, qw


def main(args=None):
    rclpy.init(args=args)
    node = IMUCalibrationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
