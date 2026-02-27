#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from shape_msgs.msg import SolidPrimitive

import tf2_ros


class TargetCollisionPublisher(Node):
    def __init__(self):
        super().__init__("target_collision_publisher")

        self.declare_parameter("planning_frame", "base_link")
        self.declare_parameter("target_frame", "Target_object")
        self.declare_parameter("object_id", "target_object")

        # Box size in meters (set to match your Isaac cube)
        self.declare_parameter("size_x", 0.10)
        self.declare_parameter("size_y", 0.10)
        self.declare_parameter("size_z", 0.10)

        # Extra safety margin around the object
        self.declare_parameter("padding", 0.02)

        self.planning_frame = self.get_parameter("planning_frame").value
        self.target_frame = self.get_parameter("target_frame").value
        self.object_id = self.get_parameter("object_id").value

        self.size_x = float(self.get_parameter("size_x").value)
        self.size_y = float(self.get_parameter("size_y").value)
        self.size_z = float(self.get_parameter("size_z").value)
        self.padding = float(self.get_parameter("padding").value)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Publish diffs to MoveIt
        self.scene_pub = self.create_publisher(PlanningScene, "/planning_scene", 10)

        self.timer = self.create_timer(0.1, self.tick)  # 10 Hz

        self.get_logger().info(
            f"Publishing collision '{self.object_id}' from TF {self.planning_frame} -> {self.target_frame} "
            f"size=({self.size_x},{self.size_y},{self.size_z}) padding={self.padding}"
        )

    def tick(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.planning_frame, self.target_frame, rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().warn(f"TF not ready: {e}")
            return

        # Build collision object
        obj = CollisionObject()
        obj.id = self.object_id
        obj.header.frame_id = self.planning_frame
        obj.operation = CollisionObject.ADD

        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [
            self.size_x + 2.0 * self.padding,
            self.size_y + 2.0 * self.padding,
            self.size_z + 2.0 * self.padding,
        ]

        pose = Pose()
        pose.position.x = tf.transform.translation.x
        pose.position.y = tf.transform.translation.y
        pose.position.z = tf.transform.translation.z
        pose.orientation = tf.transform.rotation

        obj.primitives.append(prim)
        obj.primitive_poses.append(pose)

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects.append(obj)

        self.scene_pub.publish(scene)


def main():
    rclpy.init()
    node = TargetCollisionPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
