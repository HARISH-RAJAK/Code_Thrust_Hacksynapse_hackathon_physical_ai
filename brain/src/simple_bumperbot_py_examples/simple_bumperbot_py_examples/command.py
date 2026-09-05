#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration


class JointCommandPublisher(Node):

    def __init__(self):
        super().__init__('joint_command_publisher')

        # Subscribe to /forward_kinematics topic
        self.subscription = self.create_subscription(
            Float64MultiArray,
            '/forward_kinematics',
            self.joint_angles_callback,
            10
        )

        # Publisher for arm controller
        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            '/robot1/hiwonder_xarm_controller/joint_trajectory',
            10
        )

        self.get_logger().info("Joint Command Publisher started")
        self.publish_count = 0
        self.publish_count_gripper_open = 0
        self.stored_positions = None
        self.gripper_timer = None

    def joint_angles_callback(self, msg: Float64MultiArray):
        """
        Receive joint angles from /forward_kinematics and publish to controller
        
        Input order from IK (5 joints):
        [0] base_to_Link1
        [1] Link1_to_Link2
        [2] Link2_to_Link3
        [3] Linkt_to_Link5
        [4] Link5_to_gripperbase
        
        Output order for controller (6 joints):
        [0] Link1_to_Link2
        [1] Link2_to_Link3
        [2] Link5_to_gripperbase
        [3] Linkt_to_Link5
        [4] base_to_Link1
        [5] gripper_finger1_joint
        """
        
        if len(msg.data) < 5:
            self.get_logger().warn(f"Expected 5 joint angles, got {len(msg.data)}")
            return

        # Extract IK joint angles
        base_to_link1 = msg.data[0]
        link1_to_link2 = msg.data[1]
        link2_to_link3 = msg.data[2]
        linkt_to_link5 = msg.data[3]
        link5_to_gripperbase = msg.data[4]

        # Reorder for controller
        controller_positions = [
            link1_to_link2,        # [0] Link1_to_Link2
            link2_to_link3,        # [1] Link2_to_Link3
            link5_to_gripperbase,  # [2] Link5_to_gripperbase
            linkt_to_link5,        # [3] Linkt_to_Link5
            base_to_link1,         # [4] base_to_Link1
            1.20                   # [5] gripper_finger1_joint (keep closed/neutral)
        ]

        # Create JointTrajectory message
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = [
            'Link1_to_Link2',
            'Link2_to_Link3',
            'Link5_to_gripperbase',
            'Linkt_to_Link5',
            'base_to_Link1',
            'gripper_finger1_joint'
        ]

        # Create trajectory point
        point = JointTrajectoryPoint()
        point.positions = controller_positions
        point.velocities = [0.0] * 6
        point.time_from_start = Duration(sec=3, nanosec=0)

        trajectory_msg.points = [point]
        if self.publish_count < 1:
            # Store positions for later gripper open command
            self.stored_positions = controller_positions[:5]  # Store first 5 joint positions
            
            # Publish
            self.trajectory_pub.publish(trajectory_msg)

            self.get_logger().info(
                f"Published joint trajectory: base={base_to_link1:.3f}, "
                f"L1-L2={link1_to_link2:.3f}, L2-L3={link2_to_link3:.3f}, "
                f"Lt-L5={linkt_to_link5:.3f}, L5-grip={link5_to_gripperbase:.3f}"
            )
            self.publish_count += 1
            
            # Schedule gripper open command after 10 seconds
            self.gripper_timer = self.create_timer(10.0, self.open_gripper_callback)

    def open_gripper_callback(self):
        """
        Publish the same trajectory with gripper opened (gripper_finger1_joint = 0.0)
        """
        if self.publish_count_gripper_open >= 1:
            return  # Already published
        
        if self.stored_positions is None:
            self.get_logger().warn("No stored positions to publish")
            return
        
        # Create positions with gripper open
        gripper_open_positions = self.stored_positions + [0.0]  # Append 0.0 for gripper
        
        # Create JointTrajectory message
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = [
            'Link1_to_Link2',
            'Link2_to_Link3',
            'Link5_to_gripperbase',
            'Linkt_to_Link5',
            'base_to_Link1',
            'gripper_finger1_joint'
        ]
        
        # Create trajectory point
        point = JointTrajectoryPoint()
        point.positions = gripper_open_positions
        point.velocities = [0.0] * 6
        point.time_from_start = Duration(sec=3, nanosec=0)
        
        trajectory_msg.points = [point]
        
        # Publish
        self.trajectory_pub.publish(trajectory_msg)
        self.get_logger().info("Published gripper OPEN command (gripper_finger1_joint=0.0)")
        
        self.publish_count_gripper_open += 1
        
        # Cancel timer so it doesn't fire again
        if self.gripper_timer:
            self.gripper_timer.cancel()


def main(args=None):
    rclpy.init(args=args)
    node = JointCommandPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
