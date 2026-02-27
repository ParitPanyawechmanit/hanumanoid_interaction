#!/usr/bin/env python3
"""
yolo_depth_to_tf.py

- Subscribes:
  * RGB image
  * Aligned depth image (same pixel coordinates as RGB)
  * Color CameraInfo (intrinsics)

- Runs YOLO on RGB, picks the best detection among allowed classes,
  reads depth at bbox center, back-projects to 3D (camera optical frame),
  transforms into planning frame (e.g., base_link), and broadcasts:
    planning_frame -> Target_object  (TF)

- Publishes:
  * /target_object_pose   (PoseStamped in planning_frame)
  * /target_object_size   (Vector3 box size in meters, approx from bbox+depth)
  * /yolo_debug           (debug image with bbox + depth + size overlay)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped, PoseStamped, Vector3
from cv_bridge import CvBridge

import numpy as np
import cv2

import tf2_ros
import tf2_geometry_msgs
from ultralytics import YOLO

from message_filters import Subscriber, ApproximateTimeSynchronizer


def robust_depth(depth_m: np.ndarray, u: int, v: int, win: int = 4):
    """Median depth in a small window around (u,v), ignoring zeros/NaNs."""
    h, w = depth_m.shape[:2]
    u = int(np.clip(u, 0, w - 1))
    v = int(np.clip(v, 0, h - 1))
    x1, x2 = max(0, u - win), min(w, u + win + 1)
    y1, y2 = max(0, v - win), min(h, v + win + 1)

    patch = depth_m[y1:y2, x1:x2].astype(np.float32).reshape(-1)
    patch = patch[np.isfinite(patch)]
    patch = patch[patch > 0.0]
    if patch.size == 0:
        return None
    return float(np.median(patch))


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


class YoloDepthToTF(Node):
    def __init__(self):
        super().__init__("yolo_depth_to_tf")
        self.bridge = CvBridge()

        # -------------------------
        # Parameters 
        # -------------------------
        #for realcam
        # self.declare_parameter("rgb_topic", "/camera/camera/color/image_raw")
        # self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        # self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        #for simcam
        self.declare_parameter("rgb_topic", "/hanumanoid_interaction/intelRealsense_rgb")
        self.declare_parameter("depth_topic", "/hanumanoid_interaction/intelRealsense_depth")
        self.declare_parameter("camera_info_topic", "/hanumanoid_interaction/intelRealsense_info")   

        # camframe for realcam
        # self.declare_parameter("camera_frame", "camera_color_optical_frame")
        # camframe for sim
        self.declare_parameter("camera_frame", "Camera_Pseudo_Depth")
        self.declare_parameter("planning_frame", "base_link")
        self.declare_parameter("target_frame", "Target_object")

        # self.declare_parameter("model", "yolov8n.pt")
        self.declare_parameter("model", "yolov8_custom.pt")
        self.declare_parameter("conf", 0.5)
        # self.declare_parameter("allowed_classes", ["bottle", "cup", "cell phone"])
        self.declare_parameter("allowed_classes", ["bottle", "cup", "cell phone","colorful-cube"])

        # RealSense aligned depth is commonly uint16 in millimeters.
        self.declare_parameter("depth_is_mm_uint16", True)
        self.declare_parameter("min_depth_m", 0.10)
        self.declare_parameter("max_depth_m", 3.00)

        # Box sizing options
        self.declare_parameter("size_padding_mul", 1.15)   # enlarge estimated size a bit
        self.declare_parameter("size_min_m", 0.02)         # clamp min size
        self.declare_parameter("size_max_m", 0.60)         # clamp max size
        self.declare_parameter("thickness_fallback_m", 0.06)
        self.declare_parameter("thickness_min_m", 0.02)
        self.declare_parameter("thickness_max_m", 0.30)

        # Debug image publish
        self.declare_parameter("publish_debug_image", True)

        # -------------------------
        # YOLO model
        # -------------------------
        model_path = self.get_parameter("model").value
        self.yolo = YOLO(model_path)

        # -------------------------
        # TF
        # -------------------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # -------------------------
        # Publishers
        # -------------------------
        self.pub_pose = self.create_publisher(PoseStamped, "/target_object_pose", 10)
        self.pub_size = self.create_publisher(Vector3, "/target_object_size", 10)

        self.pub_dbg = None
        if bool(self.get_parameter("publish_debug_image").value):
            self.pub_dbg = self.create_publisher(Image, "/yolo_debug", 10)

        # -------------------------
        # Camera intrinsics cache
        # -------------------------
        self.K = None  # (fx, fy, cx, cy)

        # -------------------------
        # Subscribers (sensor QoS!)
        # -------------------------
        rgb_topic = self.get_parameter("rgb_topic").value
        depth_topic = self.get_parameter("depth_topic").value
        info_topic = self.get_parameter("camera_info_topic").value

        self.sub_info = self.create_subscription(CameraInfo, info_topic, self.on_info, qos_profile_sensor_data)
        self.sub_rgb = Subscriber(self, Image, rgb_topic, qos_profile=qos_profile_sensor_data)
        self.sub_depth = Subscriber(self, Image, depth_topic, qos_profile=qos_profile_sensor_data)

        sync = ApproximateTimeSynchronizer([self.sub_rgb, self.sub_depth], queue_size=10, slop=0.08)
        sync.registerCallback(self.on_rgb_depth)

        self.get_logger().info("YOLO+Depth -> TF started.")
        self.get_logger().info(f"RGB:   {rgb_topic}")
        self.get_logger().info(f"DEPTH: {depth_topic}")
        self.get_logger().info(f"INFO:  {info_topic}")

    def on_info(self, msg: CameraInfo):
        # K = [fx 0 cx 0 fy cy 0 0 1]
        fx = msg.k[0]
        fy = msg.k[4]
        cx = msg.k[2]
        cy = msg.k[5]
        if fx > 0 and fy > 0:
            self.K = (float(fx), float(fy), float(cx), float(cy))

    def on_rgb_depth(self, rgb_msg: Image, depth_msg: Image):
        if self.K is None:
            self.get_logger().warn("Waiting for CameraInfo intrinsics...")
            return

        # Convert RGB
        bgr = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")

        # Convert depth
        depth = np.array(self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough"))

        # Depth in meters
        depth_is_mm_uint16 = bool(self.get_parameter("depth_is_mm_uint16").value)
        if depth_is_mm_uint16 and depth.dtype == np.uint16:
            depth_m = depth.astype(np.float32) * 0.001
        else:
            depth_m = depth.astype(np.float32)

        conf_th = float(self.get_parameter("conf").value)
        allowed = set(self.get_parameter("allowed_classes").value)

        # YOLO inference
        res = self.yolo(bgr, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            if self.pub_dbg:
                self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(bgr, encoding="bgr8"))
            return

        # Pick best allowed detection (highest confidence)
        best = None
        best_conf = -1.0
        for box in res.boxes:
            conf = float(box.conf[0].cpu().numpy())
            if conf < conf_th:
                continue
            cls_id = int(box.cls[0].cpu().numpy())
            label = self.yolo.names.get(cls_id, str(cls_id))
            if label not in allowed:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            if conf > best_conf:
                best_conf = conf
                best = (x1, y1, x2, y2, label, conf)

        if best is None:
            if self.pub_dbg:
                self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(bgr, encoding="bgr8"))
            return

        x1, y1, x2, y2, label, conf = best
        u = int((x1 + x2) * 0.5)
        v = int((y1 + y2) * 0.5)

        # Robust depth at bbox center
        z = robust_depth(depth_m, u, v, win=4)
        if z is None:
            self.get_logger().warn("Depth invalid at bbox center.")
            return

        min_d = float(self.get_parameter("min_depth_m").value)
        max_d = float(self.get_parameter("max_depth_m").value)
        if not (min_d <= z <= max_d):
            self.get_logger().warn(f"Depth out of range: {z:.2f} m")
            return

        fx, fy, cx, cy = self.K

        # Back-project pixel to 3D in camera optical frame
        X = (u - cx) * z / fx
        Y = (v - cy) * z / fy
        Z = z

        cam_frame = self.get_parameter("camera_frame").value
        planning = self.get_parameter("planning_frame").value
        target_frame = self.get_parameter("target_frame").value

        pose_cam = PoseStamped()
        pose_cam.header = rgb_msg.header
        pose_cam.header.frame_id = cam_frame
        pose_cam.pose.position.x = float(X)
        pose_cam.pose.position.y = float(Y)
        pose_cam.pose.position.z = float(Z)
        pose_cam.pose.orientation.w = 1.0  # identity

        # Transform to planning frame
        try:
            T = self.tf_buffer.lookup_transform(planning, cam_frame, rclpy.time.Time())
            pose_base = tf2_geometry_msgs.do_transform_pose_stamped(pose_cam, T)
        except Exception as e:
            self.get_logger().warn(f"TF failed ({planning} <- {cam_frame}): {e}")
            return

        # Broadcast Target_object TF
        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = planning
        tf_msg.child_frame_id = target_frame
        tf_msg.transform.translation.x = pose_base.pose.position.x
        tf_msg.transform.translation.y = pose_base.pose.position.y
        tf_msg.transform.translation.z = pose_base.pose.position.z
        tf_msg.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(tf_msg)

        # Publish PoseStamped for debug
        pose_base.header.stamp = tf_msg.header.stamp
        self.pub_pose.publish(pose_base)

        # -------------------------
        # Estimate object size (meters) from bbox + depth + intrinsics
        # -------------------------
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        size_x = (bw * z) / fx
        size_y = (bh * z) / fy

        # Estimate thickness from depth spread inside bbox (optional)
        roi = depth_m[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        roi = roi[np.isfinite(roi)]
        roi = roi[roi > 0.0]

        thickness_fallback = float(self.get_parameter("thickness_fallback_m").value)
        t_min = float(self.get_parameter("thickness_min_m").value)
        t_max = float(self.get_parameter("thickness_max_m").value)

        if roi.size > 200:
            z_near = float(np.percentile(roi, 10))
            z_far = float(np.percentile(roi, 90))
            size_z = clamp(z_far - z_near, t_min, t_max)
        else:
            size_z = thickness_fallback

        pad_mul = float(self.get_parameter("size_padding_mul").value)
        s_min = float(self.get_parameter("size_min_m").value)
        s_max = float(self.get_parameter("size_max_m").value)

        sx = clamp(size_x * pad_mul, s_min, s_max)
        sy = clamp(size_y * pad_mul, s_min, s_max)
        sz = clamp(size_z * pad_mul, s_min, s_max)

        size_msg = Vector3()
        size_msg.x, size_msg.y, size_msg.z = float(sx), float(sy), float(sz)
        self.pub_size.publish(size_msg)

        # -------------------------
        # Debug overlay image
        # -------------------------
        if self.pub_dbg:
            cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.circle(bgr, (u, v), 4, (0, 255, 255), -1)

            txt1 = f"{label} {conf:.2f}  z={z:.2f}m"
            txt2 = f"size ~ {sx:.2f} x {sy:.2f} x {sz:.2f} m"
            y_txt = max(20, y1 - 10)

            cv2.putText(bgr, txt1, (x1, y_txt),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(bgr, txt2, (x1, y_txt + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(bgr, encoding="bgr8"))


def main():
    rclpy.init()
    node = YoloDepthToTF()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()