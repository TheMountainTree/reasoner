#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class CameraViewer(Node):
    def __init__(self, visualize: bool = True):
        super().__init__('camera_viewer')

        self.bridge = CvBridge()
        self.visualize = visualize

        # 订阅 RGB 与 Depth 图像
        self.rgb_sub = self.create_subscription(
            Image, '/fetch_head/rgb/image_raw', self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, '/fetch_head/depth/image_raw', self.depth_callback, 10)

        # 图像缓存
        self.latest_rgb = None
        self.latest_depth = None

        self.get_logger().info("✅ CameraViewer node started, waiting for image topics...")

    def rgb_callback(self, msg: Image):
        """接收 RGB 图像并可视化"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.latest_rgb = cv_image

            if self.visualize:
                cv2.imshow("RGB Image", cv_image)
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"RGB callback error: {e}")

    def depth_callback(self, msg: Image):
        """接收 Depth 图像并可视化"""
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            self.latest_depth = cv_depth

            if self.visualize:
                # 可用伪彩色显示深度
                depth_colored = cv2.applyColorMap(cv_depth, cv2.COLORMAP_JET)
                cv2.imshow("Depth Image", depth_colored)
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Depth callback error: {e}")

    def stop(self):
        """节点退出时安全关闭"""
        self.get_logger().info("Shutting down CameraViewer...")
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = CameraViewer(visualize=True)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
