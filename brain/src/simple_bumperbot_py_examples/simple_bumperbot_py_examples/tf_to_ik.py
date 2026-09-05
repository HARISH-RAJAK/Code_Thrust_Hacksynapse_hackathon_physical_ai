#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
import ikpy.chain
import numpy as np
import tempfile
from std_msgs.msg import Float64MultiArray, String
from geometry_msgs.msg import TransformStamped

# ================== UPDATED URDF ==================
URDF_CONTENT = """<?xml version="1.0"?>
<robot name="my_robot">
    <link name="base_link"/>
    <link name="Link_1_1"/>
    <link name="Link_2_1"/>
    <link name="Link_3_1"/>
    <link name="Link_4_1"/>
    <link name="Link_5_1"/>
    <link name="tool_link"/>
    <link name="gripper_base_1"/>
    <link name="target_link"/>

    <joint name="base_to_Link1" type="revolute">
        <parent link="base_link"/>
        <child link="Link_1_1"/>
        <origin xyz="0.0 0.0 0.05157"/>
        <axis xyz="0 0 1"/>
        <limit upper="2.09" lower="-2.01"/>
    </joint>

    <joint name="Link1_to_Link2" type="revolute">
        <parent link="Link_1_1"/>
        <child link="Link_2_1"/>
        <origin xyz="-0.011 0.019 0.03"/>
        <axis xyz="0 1 0"/>
        <limit upper="1.57" lower="-1.57"/>
    </joint>

    <joint name="Link2_to_Link3" type="revolute">
        <parent link="Link_2_1"/>
        <child link="Link_3_1"/>
        <origin xyz="0 -0.0004 0.096"/>
        <axis xyz="0 1 0"/>
        <limit upper="2.35" lower="-2.35"/>
    </joint>

    <joint name="Link3_4_Bridge" type="fixed">
        <parent link="Link_3_1"/>
        <child link="Link_4_1"/>
        <origin xyz="3.8e-05 -0.019 0.058"/>
    </joint>

    <joint name="Linkt_to_Link5" type="revolute">
        <parent link="Link_4_1"/>
        <child link="Link_5_1"/>
        <origin xyz="3.8e-05 0.019 0.037"/>
        <axis xyz="0 1 0"/>
        <limit upper="1.88" lower="-1.88"/>
    </joint>

    <joint name="Link5_to_gripperbase" type="revolute">
        <parent link="Link_5_1"/>
        <child link="tool_link"/>
        <origin xyz="0.0003 -0.019 0.055" rpy="0 0 1.57"/>
        <axis xyz="0 0 1"/>
        <limit upper="1.91" lower="-1.91"/>
    </joint>

    <joint name="gripper_base_joint" type="fixed">
        <parent link="tool_link"/>
        <child link="gripper_base_1"/>
    </joint>

    <joint name="target_joint" type="fixed">
        <parent link="gripper_base_1"/>
        <child link="target_link"/>
        <origin xyz="0 0 0.07" rpy="0 0 0"/>
    </joint>

</robot>
"""


class TFToIKNode(Node):
#  for simulation         <origin xyz="0 0 0.085" rpy="0 0 0"/>

    def __init__(self):
        super().__init__('tf_to_ik_node')

        # TF2 Buffer and Listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # IK Chain setup
        with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as tmp:
            tmp.write(URDF_CONTENT)
            urdf_path = tmp.name

        self.chain = ikpy.chain.Chain.from_urdf_file(
            urdf_path,
            active_links_mask=[False, True, True, True, False, True, True, False, False]
        )

        # Publisher for joint angles (Float64MultiArray)
        self.joint_pub = self.create_publisher(
            Float64MultiArray,
            '/forward_kinematics',
            10
        )

        # Publisher for frame info (String) - to indicate which frame
        self.info_pub = self.create_publisher(
            String,
            '/forward_kinematics_info',
            10
        )

        # Timer to periodically check for frames and compute IK
        self.timer = self.create_timer(2.0, self.timer_callback)

        # Track current target
        self.current_target_index = 0
        self.target_frames = ['tube_1', 'bad_fruit_1']

        self.get_logger().info("TF to IK Node started. Looking for frames: tube_1, bad_fruit_1")

    def timer_callback(self):
        """Periodically lookup TF frames and compute IK"""
        
        # Cycle through target frames
        target_frame = self.target_frames[self.current_target_index]
        
        try:
            # Lookup transform from base_Link to target frame
            transform = self.tf_buffer.lookup_transform(
                'base_Link',  # target frame
                target_frame,  # source frame
                rclpy.time.Time(),  # get latest
                timeout=rclpy.duration.Duration(seconds=1.0)
            )

            # Extract translation (position)
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            z = transform.transform.translation.z

            self.get_logger().info(f"Found {target_frame} at position: [{x:.3f}, {y:.3f}, {z:.3f}]")

            # Compute IK
            joint_angles = self.calculate_ik([x, y, z])

            if joint_angles is not None:
                # Publish joint angles
                self.publish_joint_angles(target_frame, joint_angles)
                
                # Move to next target frame
                self.current_target_index = (self.current_target_index + 1) % len(self.target_frames)

        except Exception as e:
            self.get_logger().warn(f"Could not find frame {target_frame}: {e}")
            # Try next frame
            self.current_target_index = (self.current_target_index + 1) % len(self.target_frames)

    def calculate_ik(self, target_xyz):
        """Calculate inverse kinematics for target position"""
        
        initial_position = [0] * len(self.chain.links)

        try:
            ik_result = self.chain.inverse_kinematics(
                target_position=target_xyz,
                initial_position=initial_position
            )

            joint_angles = [
                ik_result[1],  # base_to_Link1
                ik_result[2],  # Link1_to_Link2
                ik_result[3],  # Link2_to_Link3
                ik_result[5],  # Linkt_to_Link5
                ik_result[6],  # Link5_to_gripperbase
            ]

            # Verify FK
            fk = self.chain.forward_kinematics(ik_result)
            reached_position = fk[:3, 3]
            error = np.linalg.norm(np.array(target_xyz) - reached_position)

            self.get_logger().info(f"IK Solution - Error: {error:.6f} m")

            return joint_angles

        except Exception as e:
            self.get_logger().error(f"IK calculation failed: {e}")
            return None

    def publish_joint_angles(self, frame_name, joint_angles):
        """Publish joint angles to /forward_kinematics topic"""
        
        # Publish Float64MultiArray with joint angles
        msg = Float64MultiArray()
        msg.data = joint_angles
        self.joint_pub.publish(msg)

        # Publish info message with frame name and angles
        info_msg = String()
        angles_str = ", ".join([f"{angle:.6f}" for angle in joint_angles])
        info_msg.data = f'["{frame_name}", {angles_str}]'
        self.info_pub.publish(info_msg)

        self.get_logger().info(f'Published IK for {frame_name}: [{angles_str}]')


def main(args=None):
    rclpy.init(args=args)
    node = TFToIKNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
