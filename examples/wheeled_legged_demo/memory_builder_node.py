"""
MemoryBuilderNode for wheeled-legged robot.
Subscribes to /caption and /amcl_pose, builds memory in Milvus Lite.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import String
import math
import numpy as np

from lite_memory import LiteMemory, MemoryItem


def pose_msg_to_values(msg: PoseWithCovarianceStamped):
    """Extract position [x,y,z], yaw angle, and timestamp from a PoseWithCovarianceStamped."""
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    position = [p.x, p.y, p.z]

    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    stamp = msg.header.stamp
    pose_time = float(stamp.sec) + float(stamp.nanosec) * 1e-9

    return position, yaw, pose_time


class MemoryBuilderNode(Node):

    def __init__(self):
        super().__init__("MemoryBuilderNode")

        self.declare_parameter("db_path", "./remembr_memory.db")
        self.declare_parameter("db_collection", "robot_memory")
        self.declare_parameter("ollama_host", "192.168.31.63")
        self.declare_parameter("ollama_port", 11434)
        self.declare_parameter("pose_topic", "/amcl_pose")
        self.declare_parameter("caption_topic", "/caption")

        self.memory = LiteMemory(
            db_path=self.get_parameter("db_path").value,
            collection_name=self.get_parameter("db_collection").value,
            ollama_host=self.get_parameter("ollama_host").value,
            ollama_port=self.get_parameter("ollama_port").value,
        )

        self.pose_subscriber = self.create_subscription(
            PoseWithCovarianceStamped,
            self.get_parameter("pose_topic").value,
            self.pose_callback,
            10,
        )
        self.caption_subscriber = self.create_subscription(
            String,
            self.get_parameter("caption_topic").value,
            self.caption_callback,
            10,
        )

        self.pose_msg = None
        self.logger = self.get_logger()
        self.logger.info(
            f"MemoryBuilder ready. DB: {self.get_parameter('db_path').value}, "
            f"Collection: {self.get_parameter('db_collection').value}"
        )

    def pose_callback(self, msg: PoseWithCovarianceStamped):
        self.pose_msg = msg

    def caption_callback(self, msg: String):
        if self.pose_msg is None:
            self.logger.warn("Received caption but no pose yet, skipping")
            return

        try:
            position, angle, pose_time = pose_msg_to_values(self.pose_msg)

            caption_text = msg.data
            image_path = ""
            if caption_text.startswith("[IMG:"):
                end = caption_text.index("]")
                image_path = caption_text[5:end]
                caption_text = caption_text[end + 1:]

            item = MemoryItem(
                caption=caption_text,
                time=pose_time,
                position=position,
                theta=angle,
                image_path=image_path,
            )
            self.memory.insert(item)

            count = self.memory.count()
            img_note = f", img={image_path}" if image_path else ""
            self.logger.info(
                f"Memory #{count}: pos={np.array(position).round(2).tolist()}{img_note}, "
                f"caption={caption_text[:80]}..."
            )
        except Exception as e:
            self.logger.error(f"Failed to insert memory: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = MemoryBuilderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
