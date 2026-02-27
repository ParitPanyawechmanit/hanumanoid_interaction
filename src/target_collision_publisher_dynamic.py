#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Vector3
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose

import tf2_ros
import numpy as np


class TargetCollisionPublisherDynamic(Node):
    def __init__(self):
        super().__init__("target_collision_publisher_dynamic")

        # Params compatible with your current usage
        self.declare_parameter("planning_frame", "base_link")
        self.declare_parameter("target_frame", "Target_object")
        self.declare_parameter("object_id", "target_object")

        # If size topic not received yet, fallback size:
        self.declare_parameter("default_size_x", 0.10)
        self.declare_parameter("default_size_y", 0.10)
        self.declare_parameter("default_size_z", 0.10)

        # Extra padding added (meters)
        self.declare_parameter("padding", 0.02)

        # Size topic
        self.declare_parameter("size_topic", "/target_object_size")

        # publish rate
        self.declare_parameter("rate_hz", 10.0)

        self.planning_frame = self.get_parameter("planning_frame").value
        self.target_frame = self.get_parameter("target_frame").value
        self.object_id = self.get_parameter("object_id").value
        self.padding = float(self.get_parameter("padding").value)

        self.size = np.array([
            float(self.get_parameter("default_size_x").value),
            float(self.get_parameter("default_size_y").value),
            float(self.get_parameter("default_size_z").value),
        ], dtype=np.float32)

        # Smooth size to reduce jitter
        self.alpha = 0.30  # EMA smoothing

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # MoveIt listens to /collision_object
        self.pub = self.create_publisher(CollisionObject, "/collision_object", 10)

        # Subscribe size updates
        size_topic = self.get_parameter("size_topic").value
        self.create_subscription(Vector3, size_topic, self.on_size, 10)

        rate_hz = float(self.get_parameter("rate_hz").value)
        self.timer = self.create_timer(1.0 / max(1e-3, rate_hz), self.on_timer)

        self.get_logger().info(f"Publishing collision box '{self.object_id}' at TF '{self.target_frame}' in '{self.planning_frame}'")
        self.get_logger().info(f"Listening size on: {size_topic}")

    def on_size(self, msg: Vector3):
        new = np.array([msg.x, msg.y, msg.z], dtype=np.float32)

        # Clamp (avoid crazy spikes)
        new = np.clip(new, 0.02, 1.00)

        # Smooth
        self.size = (1.0 - self.alpha) * self.size + self.alpha * new

    def on_timer(self):
        try:
            T = self.tf_buffer.lookup_transform(self.planning_frame, self.target_frame, rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn(f"TF not ready: {e}")
            return

        # Pose from TF
        p = Pose()
        p.position.x = T.transform.translation.x
        p.position.y = T.transform.translation.y
        p.position.z = T.transform.translation.z
        p.orientation.w = 1.0

        sx, sy, sz = (self.size + self.padding).tolist()

        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [float(sx), float(sx), float(sy)]

        co = CollisionObject()
        co.header.stamp = self.get_clock().now().to_msg()
        co.header.frame_id = self.planning_frame
        co.id = self.object_id
        co.primitives = [prim]
        co.primitive_poses = [p]
        co.operation = CollisionObject.ADD

        self.pub.publish(co)


def main():
    rclpy.init()
    node = TargetCollisionPublisherDynamic()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()