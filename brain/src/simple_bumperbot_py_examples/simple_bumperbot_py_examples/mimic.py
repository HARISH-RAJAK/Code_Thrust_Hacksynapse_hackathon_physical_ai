import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

class GripperMimic(Node):
    def __init__(self):
        super().__init__('gripper_mimic')

        self.sub = self.create_subscription(
            Float64,
            '/joint5/cmd_pos',
            self.cb,
            10)

        self.pub = self.create_publisher(
            Float64,
            '/joint6/cmd_pos',
            10)

    def cb(self, msg):
        mimic = Float64()
        mimic.data = -msg.data   # 🔥 mimic logic
        self.pub.publish(mimic)

def main():
    rclpy.init()
    node = GripperMimic()
    rclpy.spin(node)
    rclpy.shutdown()
if __name__ == '__main__':
    main()