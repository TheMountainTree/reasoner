#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters
import os
import json
import time
from datetime import datetime
import torch

import cv2
import numpy as np
import torch
from sensor_msgs.msg import Image
from PIL import Image as PIL_Image
import supervision as sv
import open3d as o3d
import os
from reasoner.sam2.build_sam import build_sam2
from reasoner.sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForCausalLM
from pics_segmentation_lib import grounded_sam2_detection
# -----------------------------



# -----------------------------

class RGBDProcessorNode(Node):
    def __init__(self):
        super().__init__('rgbd_processor_node')

        # 创建 CvBridge
        self.bridge = CvBridge()

        # 状态变量
        self.processing_enabled = False
        self.tmp_base_dir = os.path.join(os.getcwd(), 'tmp')  # 项目目录下的 tmp

        # --- 订阅 state topic ---
        state_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.state_sub = self.create_subscription(
            String,
            '/task_state',
            self.state_callback,
            state_qos
        )

        # --- 同步订阅 RGB + Depth ---
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.rgb_sub = message_filters.Subscriber(
            self, Image, '/fetch_head/rgb/image_raw', qos_profile=image_qos
        )
        self.depth_sub = message_filters.Subscriber(
            self, Image, '/fetch_head/depth/image_raw', qos_profile=image_qos
        )

        # 使用时间同步器（可根据需要改用 ApproximateTimeSynchronizer）
        self.ts = message_filters.TimeSynchronizer([self.rgb_sub, self.depth_sub], 10)
        self.ts.registerCallback(self.image_callback)

        self.get_logger().info("RGBD Processor Node started. Waiting for state='D'...")

    def state_callback(self, msg: String):
        state = msg.data.strip()
        if state == 'D':
            self.processing_enabled = True
            self.get_logger().info("State 'D' received. Next RGBD pair will be processed.")
        else:
            self.processing_enabled = False
            self.get_logger().info(f"State changed to '{state}'. Processing disabled.")

    def image_callback(self, rgb_msg: Image, depth_msg: Image):
        if not self.processing_enabled:
            return  # 不处理

        try:
            # 转换为 OpenCV 图像
            rgb_cv = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth_cv = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Failed to convert images: {e}")
            return

        # 调用处理函数
        try:
            rgb_results = self.process_rgb_image(rgb_cv)  # 每次处理n张对象图
        except Exception as e:
            self.get_logger().error(f"Image processing failed: {e}")
            return

        # 保存 JSON
        json_path = os.path.join(self.tmp_base_dir, "relation_graph.json")  # 命名固定
        try:
            with open(json_path, 'w') as f:
                json.dump(rgb_results, f, indent=2)
            self.get_logger().info(f"Saved relation graph to {json_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to save JSON: {e}")

        # 关闭处理，等待下一次手动开启
        self.processing_enabled = False
        self.get_logger().info("Processing completed. Enabled set to False.")

    def process_rgb_image(self, rgb_cv):
        """
        对单帧图像执行 Florence2 + SAM2 分割
        """
        # ----------------------
        # 1. 初始化模型
        # ----------------------
        FLORENCE2_MODEL_ID = "/home/mlin/project_erobot/florence2"
        SAM2_CHECKPOINT = "/home/mlin/project_erobot/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
        SAM2_CONFIG = "//home/mlin/eegbot_ws/src/robot_ctr/reasoner/reasoner/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Florence2
        florence2_model = AutoModelForCausalLM.from_pretrained(
            FLORENCE2_MODEL_ID, trust_remote_code=True, torch_dtype='auto'
        ).eval().to(device)
        florence2_processor = AutoProcessor.from_pretrained(
            FLORENCE2_MODEL_ID, trust_remote_code=True
        )

        # SAM2
        sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
        sam2_predictor = SAM2ImagePredictor(sam2_model)

        rgb_pil = PIL_Image.fromarray(rgb_cv)

        detection_results = grounded_sam2_detection(
            florence2_model,
            florence2_processor,
            sam2_predictor,
            rgb_pil,
            visualize=False,
            output_dir=None
        )

        # ----------------------
        # 2. 保存对象图像
        # ----------------------
        output_dir = './segmented_objects'
        os.makedirs(output_dir, exist_ok=True)

        image = detection_results['image']
        boxes = detection_results['boxes']
        masks = detection_results['masks']
        class_names = detection_results['class_names']
        class_ids = detection_results['class_ids']
        scores = detection_results['scores']

        for i, (box, mask, class_name, class_id, score) in enumerate(zip(boxes, masks, class_names, class_ids, scores)):
            x1, y1, x2, y2 = map(int, box)
            object_image = image[y1:y2, x1:x2].copy()

            mask_in_object_area = mask[y1:y2, x1:x2].astype(bool)
            # mask_in_object_area = mask[y1:y2, x1:x2]
            object_image[~mask_in_object_area] = 0

            # 命名固定，保证覆盖
            filename = f"{class_name}_id{class_id}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, object_image)

        # 返回 JSON 数据

        rgb_results = {
            "entity_names": list(class_names),
            "entity_ids": [int(i) for i in class_ids],           # int64 -> int
            "scores": [float(s) for s in scores]                # float32/float64 -> float
        }
        return rgb_results


def main(args=None):
    rclpy.init(args=args)
    node = RGBDProcessorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()