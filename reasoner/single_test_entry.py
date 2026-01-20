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
import torch
import cv2
import numpy as np
from PIL import Image as PIL_Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForCausalLM
from pics_segmentation_lib import grounded_sam2_detection


class RGBDProcessorNode(Node):
    def __init__(self):
        super().__init__('rgbd_processor_node')
        self.bridge = CvBridge()
        self.processing_enabled = False
        self.tmp_base_dir = os.path.join(os.getcwd(), 'tmp')
        os.makedirs(self.tmp_base_dir, exist_ok=True)

        self.camera_intrinsics = np.array([
            [1.0599465578241038e+03, 0., 9.5488326677588441e+02],
            [0., 1.0539326808799726e+03, 5.2373858291060583e+02],
            [0., 0., 1.]
        ])
        # --- 初始化模型（只加载一次！）---
        self.get_logger().info("Loading Florence2 and SAM2 models...")
        FLORENCE2_MODEL_ID = "/home/mlin/project_erobot/florence2"
        SAM2_CHECKPOINT = "/home/mlin/project_erobot/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
        SAM2_CONFIG = "//home/mlin/eegbot_ws/src/robot_ctr/reasoner/reasoner/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.florence2_model = AutoModelForCausalLM.from_pretrained(
            FLORENCE2_MODEL_ID, trust_remote_code=True, torch_dtype='auto'
        ).eval().to(device)
        self.florence2_processor = AutoProcessor.from_pretrained(
            FLORENCE2_MODEL_ID, trust_remote_code=True
        )
        sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)
        self.get_logger().info("Models loaded successfully.")

        # --- 订阅 ---
        state_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.state_sub = self.create_subscription(String, '/task_state', self.state_callback, state_qos)

        image_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        # self.rgb_sub = message_filters.Subscriber(self, Image, '/kinect2/hd/image_color', qos_profile=image_qos)
        # self.depth_sub = message_filters.Subscriber(self, Image, '/kinect2/hd/image_depth_rect', qos_profile=image_qos)
        self.rgb_sub = message_filters.Subscriber(self, Image, '/fetch_head/rgb/image_raw', qos_profile=image_qos)
        self.depth_sub = message_filters.Subscriber(self, Image, '/fetch_head/depth/image_raw', qos_profile=image_qos)
        self.ts = message_filters.TimeSynchronizer([self.rgb_sub, self.depth_sub], 10)
        self.ts.registerCallback(self.image_callback)

        self.get_logger().info("RGBD Processor Node started. Waiting for state='D'...")

    def state_callback(self, msg: String):
        state = msg.data.strip()
        self.processing_enabled = (state == 'D')
        self.get_logger().info(f"State {'enabled' if self.processing_enabled else 'disabled'}: '{state}'")


    def image_callback(self, rgb_msg, depth_msg):
        if not self.processing_enabled:
            return

        # === ROS → OpenCV ===
        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough').astype(np.float32)
        depth /= 1000.0  # mm → m（如果是 Kinect）

        H, W = depth.shape

        # === 保存 RGB ===
        rgb_path = os.path.join(self.tmp_base_dir, "rgb.png")
        cv2.imwrite(rgb_path, rgb)

        # === Florence2 + SAM2 ===
        rgb_pil = PIL_Image.fromarray(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        det = grounded_sam2_detection(
            self.florence2_model,
            self.florence2_processor,
            self.sam2_predictor,
            rgb_pil,
            visualize=False
        )

        # === ReKep-style seg ===
        seg = np.zeros((H, W), dtype=np.int32)
        obj_id_to_name = {}

        for i, (mask, name) in enumerate(zip(det['masks'], det['class_names'])):
            obj_id = i + 1
            seg[mask.astype(bool)] = obj_id
            obj_id_to_name[obj_id] = name

        # === points ===
        points = self.depth_to_points(depth, self.camera_intrinsics)

        # === 保存 ===
        np.save(os.path.join(self.tmp_base_dir, "seg.npy"), seg)
        np.save(os.path.join(self.tmp_base_dir, "points.npy"), points)

        with open(os.path.join(self.tmp_base_dir, "obj_id_to_name.json"), "w") as f:
            json.dump(obj_id_to_name, f, indent=2)

        self.processing_enabled = False
        self.get_logger().info("Saved RGB / seg / points (ReKep-compatible)")
    

    def depth_to_points(self, depth, cam_K):
        """
        depth: (H, W) float32, meters
        cam_K: (3,3)
        return: (H, W, 3) float32
        """
        fx, fy = cam_K[0, 0], cam_K[1, 1]
        cx, cy = cam_K[0, 2], cam_K[1, 2]

        H, W = depth.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))

        Z = depth
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        points = np.stack([X, Y, Z], axis=-1).astype(np.float32)
        points[Z <= 0] = 0
        return points


    def process_rgb_image(self, rgb_cv):
        """
        返回 ReKep 兼容格式：
          - seg_mask: (H, W) int32 array, 0=background, 1..N=object IDs
          - obj_id_to_name: {1: "apple", 2: "cup", ...}
          - metadata: 原始检测信息（用于调试）
        """
        rgb_pil = PIL_Image.fromarray(cv2.cvtColor(rgb_cv, cv2.COLOR_BGR2RGB))

        detection_results = grounded_sam2_detection(
            self.florence2_model,
            self.florence2_processor,
            self.sam2_predictor,
            rgb_pil,
            visualize=False,
            output_dir=None
        )

        H, W = rgb_cv.shape[:2]
        seg_mask = np.zeros((H, W), dtype=np.int32)  # ReKep expects int-like mask
        obj_id_to_name = {}
        detected_objects = []

        boxes = detection_results['boxes']
        masks = detection_results['masks']
        class_names = detection_results['class_names']
        scores = detection_results['scores']

        for idx, (mask, class_name, score) in enumerate(zip(masks, class_names, scores)):
            obj_id = idx + 1  # ReKep-style: start from 1
            seg_mask[mask.astype(bool)] = obj_id
            obj_id_to_name[obj_id] = class_name
            detected_objects.append({
                "id": obj_id,
                "name": class_name,
                "score": float(score)
            })

        # 保存裁剪对象图（保留你原来的功能）
        output_dir = './segmented_objects'
        os.makedirs(output_dir, exist_ok=True)
        image_rgb = cv2.cvtColor(rgb_cv, cv2.COLOR_BGR2RGB)  # grounded_sam2 uses RGB
        for idx, (box, mask, class_name) in enumerate(zip(boxes, masks, class_names)):
            x1, y1, x2, y2 = map(int, box)
            obj_img = image_rgb[y1:y2, x1:x2].copy()
            obj_mask = mask[y1:y2, x1:x2].astype(bool)
            obj_img[~obj_mask] = 0
            filename = f"{class_name}_id{idx+1}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), cv2.cvtColor(obj_img, cv2.COLOR_RGB2BGR))

        metadata = {
            "entity_names": class_names,
            "entity_ids": list(range(1, len(class_names)+1)),
            "scores": [float(s) for s in scores]
        }

        return seg_mask, obj_id_to_name, metadata


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