import cv2
import numpy as np
import torch
from PIL import Image
import supervision as sv
import open3d as o3d
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForCausalLM


def run_florence2(task_prompt, text_input, model, processor, image):
    assert model is not None, "You should pass the init florence-2 model here"
    assert processor is not None, "You should set florence-2 processor here"

    device = model.device

    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"].to(device),
        pixel_values=inputs["pixel_values"].to(device),
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    return parsed_answer


def grounded_sam2_detection(
        florence2_model,
        florence2_processor,
        sam2_predictor,
        image_path,
        task_prompt="<OD>",
        text_input=None,
        visualize=False,
        output_dir=None
):
    """
    使用Grounded-SAM2进行物体检测与分割

    参数:
        florence2_model: Florence2模型
        florence2_processor: Florence2处理器
        sam2_predictor: SAM2预测器
        image_path: 输入图像路径
        task_prompt: 任务提示符 (默认为"<OD>"表示目标检测)
        text_input: 文本输入 (可选)
        visualize: 是否可视化结果
        output_dir: 输出目录 (用于保存可视化结果)

    返回:
        {
            'image': 原始图像(numpy数组),
            'boxes': 检测框列表[[x1, y1, x2, y2], ...],
            'masks': 分割掩码列表(布尔数组),
            'class_names': 类别名称列表,
            'class_ids': 类别ID列表,
            'scores': 检测分数列表
        }
    """
    assert text_input is None, "Text input should be None when calling dense region caption pipeline."

    # 加载图像
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # 运行Florence2目标检测
    results = run_florence2(
        task_prompt, text_input, florence2_model, florence2_processor, image
    )[task_prompt]

    # 解析检测结果
    input_boxes = np.array(results["bboxes"])
    class_names = results["labels"]
    class_ids = np.array(list(range(len(class_names))))
    scores = np.ones(len(class_names))  # 假设所有检测置信度为1

    # 使用SAM2预测分割掩码
    sam2_predictor.set_image(image_np)
    masks, _, _ = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    # 调整掩码维度
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    # 可视化结果
    if visualize:
        img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        detections = sv.Detections(
            xyxy=input_boxes,
            mask=masks.astype(bool),
            class_id=class_ids,
            confidence=scores
        )

        # 创建带标签的标注图像
        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

        label_annotator = sv.LabelAnnotator()
        labels = [f"{name}" for name in class_names]
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )

        # 创建带掩码的标注图像
        mask_annotator = sv.MaskAnnotator()
        masked_frame = mask_annotator.annotate(scene=img.copy(), detections=detections)

        # 保存结果
        if output_dir:
            cv2.imwrite(os.path.join(output_dir, "detection_annotated.jpg"), annotated_frame)
            cv2.imwrite(os.path.join(output_dir, "mask_annotated.jpg"), masked_frame)

    # 返回结果
    return {
        'image': image_np,
        'boxes': input_boxes,
        'masks': masks,
        'class_names': class_names,
        'class_ids': class_ids,
        'scores': scores
    }


def generate_object_pointclouds(
        rgb_image,
        depth_image,
        camera_intrinsics,
        detection_results,
        min_points=100,
        visualize=False
):
    """
    基于检测结果和深度图生成3D实体点云

    参数:
        rgb_image: RGB图像 (H, W, 3)
        depth_image: 深度图像 (H, W) 单位:米
        camera_intrinsics: 相机内参矩阵 (3x3)
        detection_results: 检测结果 (来自grounded_sam2_detection)
        min_points: 最小点数阈值
        visualize: 是否可视化点云

    返回:
        list: 每个检测对象的点云列表
    """
    # 提取相机内参
    fx, fy, cx, cy = camera_intrinsics[0, 0], camera_intrinsics[1, 1], camera_intrinsics[0, 2], camera_intrinsics[1, 2]

    # 准备结果容器
    object_pointclouds = []

    # 处理每个检测对象
    for i in range(len(detection_results['boxes'])):
        mask = detection_results['masks'][i]
        class_name = detection_results['class_names'][i]

        # 创建点云
        pcd = create_object_pcd(
            depth_image, mask, camera_intrinsics, rgb_image
        )

        # 检查点云有效性
        if len(pcd.points) < min_points:
            print(f"跳过 {class_name} - 点数不足: {len(pcd.points)} < {min_points}")
            continue

        # 添加元数据
        # pcd.metadata = {
        #     'class_name': class_name,
        #     'class_id': detection_results['class_ids'][i],
        #     'score': detection_results['scores'][i],
        #     'box': detection_results['boxes'][i],
        #     'num_points': len(pcd.points)
        # }
        object_info = {
            'pointcloud': pcd,
            'class_name': class_name,
            'class_id': detection_results['class_ids'][i],
            'score': detection_results['scores'][i],
            'box': detection_results['boxes'][i],
            'num_points': len(pcd.points)
        }
        object_pointclouds.append(object_info)

        # 可视化点云
        if visualize:
            print(f"可视化 {class_name} 点云 ({len(pcd.points)} 点)")
            o3d.visualization.draw_geometries([pcd])

    return object_pointclouds


def create_object_pcd(depth_image, mask, cam_K, rgb_image):
    """
    从深度图和掩码创建单个对象的点云

    参数:
        depth_image: 深度图像 (H, W) 单位:米
        mask: 对象掩码 (H, W) 布尔类型
        cam_K: 相机内参矩阵 (3x3)
        rgb_image: RGB图像 (H, W, 3)

    返回:
        o3d.geometry.PointCloud: 对象的点云
    """
    # 提取相机内参
    fx, fy, cx, cy = cam_K[0, 0], cam_K[1, 1], cam_K[0, 2], cam_K[1, 2]

    # 过滤无效深度
    valid_mask = np.logical_and(mask, depth_image > 0)

    # 如果没有有效点，返回空点云
    if not np.any(valid_mask):
        return o3d.geometry.PointCloud()

    # 获取有效点的坐标
    height, width = depth_image.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    u = u[valid_mask]
    v = v[valid_mask]
    z = depth_image[valid_mask]

    # 转换为3D坐标 (相机坐标系)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # 创建点云
    points = np.stack((x, y, z), axis=-1)

    # 获取颜色
    colors = rgb_image[valid_mask].astype(np.float32) / 255.0

    # 创建Open3D点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors[:, [2, 1, 0]])  # RGB to BGR for Open3D

    # === 新增：坐标系转换 ===
    # 创建旋转矩阵：绕X轴旋转180度 (从相机坐标系到标准坐标系)
    # 这将使Y轴向上，Z轴向前
    # 0118 0956 WLY modifed
    # R = pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0))
    # pcd.rotate(R, center=(0, 0, 0))

    # 可选: 降采样和去噪
    if len(points) > 1000:
        pcd = pcd.voxel_down_sample(voxel_size=0.005)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    return pcd

def make_S_flip_x():
    S = np.eye(4)
    S[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz((np.pi, 0, 0))
    return S


def calculate_centroid_distances(object_clouds):
    """
    计算各个实体点云质心之间的距离矩阵

    参数:
        object_clouds: 实体点云列表，每个元素是包含'pointcloud'键的字典

    返回:
        dist_matrix: 距离矩阵 (N x N)，其中N是实体数量
        centroids: 每个实体的质心坐标列表
    """
    # 计算每个实体的质心
    centroids = []
    for obj_info in object_clouds:
        pcd = obj_info['pointcloud']
        if pcd and len(pcd.points) > 0:
            centroid = np.mean(np.asarray(pcd.points), axis=0)
            centroids.append(centroid)
        else:
            centroids.append(None)  # 如果没有点云

    # 初始化距离矩阵
    n = len(centroids)
    dist_matrix = np.full((n, n), np.nan)  # 使用NaN填充无效值

    # 计算两两之间的距离
    for i in range(n):
        if centroids[i] is None:
            continue

        for j in range(i + 1, n):
            if centroids[j] is None:
                continue

            # 计算欧氏距离
            distance = np.linalg.norm(centroids[i] - centroids[j])
            dist_matrix[i, j] = distance
            dist_matrix[j, i] = distance  # 对称矩阵

    return dist_matrix, centroids


def visualize_distance_matrix(dist_matrix, class_names):
    """
    可视化距离矩阵

    参数:
        dist_matrix: 距离矩阵
        class_names: 实体类别名称列表
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm

    fig, ax = plt.subplots(figsize=(10, 8))

    # 创建热力图
    cax = ax.matshow(dist_matrix, cmap=cm.viridis)
    fig.colorbar(cax, label='距离 (米)')

    # 设置坐标轴标签
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='left')
    ax.set_yticklabels(class_names)

    # 添加距离数值
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if not np.isnan(dist_matrix[i, j]):
                ax.text(j, i, f'{dist_matrix[i, j]:.2f}',
                        ha='center', va='center', color='w')

    plt.title('实体间质心距离矩阵')
    plt.tight_layout()
    plt.savefig('output/distance_matrix.png')
    plt.show()


def calculate_bounding_boxes(object_clouds, safety_margin=0.05):
    """
    为每个点云计算安全边界框

    参数:
        object_clouds: 实体点云列表
        safety_margin: 安全裕度 (米) - 边界框在每个方向上的扩展量

    返回:
        更新后的object_clouds列表，每个对象添加了边界框信息
    """
    for obj_info in object_clouds:
        pcd = obj_info['pointcloud']
        if len(pcd.points) == 0:
            continue

        # 计算点云的轴对齐边界框 (AABB)
        aabb = pcd.get_axis_aligned_bounding_box()
        aabb.color = (1, 0, 0)  # 红色表示AABB

        # 计算点云的方向包围盒 (OBB) - 最小体积包围盒
        obb = pcd.get_oriented_bounding_box()
        obb.color = (0, 1, 0)  # 绿色表示OBB

        # 计算带安全裕度的边界框 (用于避障)
        # 使用OBB作为基础，因为通常更紧凑
        center = obb.center
        extent = np.asarray(obb.extent) + safety_margin * 2  # 每个方向增加安全裕度

        # 创建带安全裕度的边界框
        safety_box = o3d.geometry.OrientedBoundingBox(center, obb.R, extent)
        safety_box.color = (0, 0, 1)  # 蓝色表示安全边界框

        # 提取边界框参数
        # 方向向量 (R矩阵的列向量)
        orientation = np.asarray(obb.R)
        u = orientation[:, 0]  # X轴方向
        v = orientation[:, 1]  # Y轴方向
        w = orientation[:, 2]  # Z轴方向

        # 计算边界框的8个角点
        corners = np.asarray(safety_box.get_box_points())

        # 存储边界框信息
        obj_info['bounding_boxes'] = {
            'aabb': aabb,
            'obb': obb,
            'safety_box': safety_box,
            'center': center,
            'extent': extent,
            'orientation': orientation,
            'corners': corners
        }

        # 存储边界框尺寸信息
        obj_info['bbox_dimensions'] = {
            'length': extent[0],  # 最长尺寸
            'width': extent[1],  # 中等尺寸
            'height': extent[2],  # 最短尺寸
            'volume': extent[0] * extent[1] * extent[2]
        }

    return object_clouds


def visualize_bounding_boxes(object_clouds, output_dir="output"):
    """
    可视化点云及其边界框

    参数:
        object_clouds: 实体点云列表
        output_dir: 输出目录
    """
    # 为每个对象创建单独的可视化
    for i, obj_info in enumerate(object_clouds):
        if 'bounding_boxes' not in obj_info:
            continue

        pcd = obj_info['pointcloud']
        bboxes = obj_info['bounding_boxes']

        # 创建可视化对象列表
        vis_objects = [pcd]

        # 添加坐标轴
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        vis_objects.append(coord_frame)

        # 添加边界框
        vis_objects.append(bboxes['aabb'])
        vis_objects.append(bboxes['obb'])
        vis_objects.append(bboxes['safety_box'])

        # 设置视图选项
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1200, height=800)

        for obj in vis_objects:
            vis.add_geometry(obj)

        # 设置视角
        view_ctl = vis.get_view_control()
        view_ctl.set_front([0, -1, 0])  # 从侧面看
        view_ctl.set_up([0, 0, 1])  # Z轴向上

        # 渲染并保存图像
        vis.run()

        # 保存截图
        image_path = os.path.join(output_dir, f"bbox_{obj_info['class_name']}_{i}.png")
        vis.capture_screen_image(image_path)

        vis.destroy_window()

    # 创建所有对象和边界框的全局可视化
    global_vis_objects = []

    # 添加坐标轴
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    global_vis_objects.append(coord_frame)

    # 添加所有点云和边界框
    for obj_info in object_clouds:
        if 'bounding_boxes' not in obj_info:
            continue

        global_vis_objects.append(obj_info['pointcloud'])
        global_vis_objects.append(obj_info['bounding_boxes']['safety_box'])  # 只显示安全边界框

    # 全局可视化
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1600, height=900)

    for obj in global_vis_objects:
        vis.add_geometry(obj)

    # 设置视角
    view_ctl = vis.get_view_control()
    view_ctl.set_front([0, -1, 0.5])  # 从侧面看
    view_ctl.set_up([0, 0, 1])  # Z轴向上

    # 渲染并保存图像
    vis.run()

    # 保存截图
    global_image_path = os.path.join(output_dir, "global_bbox_visualization.png")
    vis.capture_screen_image(global_image_path)

    vis.destroy_window()


def save_bounding_box_info(object_clouds, output_dir="output"):
    """
    保存边界框信息到文本文件

    参数:
        object_clouds: 实体点云列表
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    info_path = os.path.join(output_dir, "bounding_box_info.txt")

    with open(info_path, 'w') as f:
        f.write("物体安全边界框信息 (用于避障)\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'物体名称':<20}{'中心位置(x,y,z)':<30}{'尺寸(长,宽,高)':<25}{'体积':<10}\n")
        f.write("-" * 60 + "\n")

        for obj_info in object_clouds:
            if 'bbox_dimensions' not in obj_info:
                continue

            dim = obj_info['bbox_dimensions']
            center = obj_info['bounding_boxes']['center']

            f.write(f"{obj_info['class_name']:<20}")
            f.write(f"({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})  ")
            f.write(f"{dim['length']:.3f}x{dim['width']:.3f}x{dim['height']:.3f} m  ")
            f.write(f"{dim['volume']:.3f} m³\n")

        f.write("\n\n边界框角点坐标 (安全边界):\n")
        f.write("=" * 60 + "\n")

        for obj_info in object_clouds:
            if 'bounding_boxes' not in obj_info:
                continue

            corners = obj_info['bounding_boxes']['corners']
            f.write(f"\n物体: {obj_info['class_name']}\n")
            f.write("-" * 60 + "\n")

            for i, corner in enumerate(corners):
                f.write(f"角点 {i + 1}: ({corner[0]:.3f}, {corner[1]:.3f}, {corner[2]:.3f})\n")

# 01 16 09:45 WLY add
def save_instance_masks(
    detection_results,
    output_dir,
    prefix="mask"
):
    """
    将 detection_results 中的实例掩码按 类别+编号 保存为文件

    输出示例:
        person_1.png
        person_1.npy
        person_2.png
        chair_1.png
        table_1.png

    参数:
        detection_results: grounded_sam2_detection 的返回值
        output_dir: 输出目录
        prefix: 文件名前缀
    """
    os.makedirs(output_dir, exist_ok=True)

    masks = detection_results["masks"]
    class_names = detection_results["class_names"]

    # 用于给同类物体编号
    class_counter = {}

    for i, (mask, cls_name) in enumerate(zip(masks, class_names)):
        # 统计同类编号
        if cls_name not in class_counter:
            class_counter[cls_name] = 1
        else:
            class_counter[cls_name] += 1

        instance_id = class_counter[cls_name]

        # 统一命名：person_1 / person_2 ...
        base_name = f"{cls_name}_{instance_id}"

        # ---------- 保存为 PNG（可视化 / 调试） ----------
        mask_img = (mask.astype(np.uint8) * 255)
        png_path = os.path.join(output_dir, f"{prefix}_{base_name}.png")
        cv2.imwrite(png_path, mask_img)

        # ---------- 保存为 NPY（精确反向映射） ----------
        npy_path = os.path.join(output_dir, f"{prefix}_{base_name}.npy")
        np.save(npy_path, mask)

        print(f"[Saved] {base_name}: {png_path}, {npy_path}")


# Open-Vocabulary Detection + Segmentation
def open_vocabulary_detection_and_segmentation(
    florence2_model,
    florence2_processor,
    sam2_predictor,
    image_path,
    task_prompt="<OPEN_VOCABULARY_DETECTION>",
    text_input=None,
    visualize=False,
    output_dir=None
):
    """
    Open-Vocabulary Detection + SAM2 Segmentation
    输出格式严格对齐 grounded_sam2_detection
    """

    assert text_input is not None, \
        "Text input should not be None when calling open-vocabulary detection pipeline."

    # ---------- 1. 读图 ----------
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # ---------- 2. Florence-2 ----------
    results = run_florence2(
        task_prompt,
        text_input,
        florence2_model,
        florence2_processor,
        image
    )[task_prompt]

    # ---------- 3. 解析检测结果 ----------
    input_boxes = np.array(results["bboxes"], dtype=np.float32)

    # Florence-2 open-vocab 使用的是 bboxes_labels
    class_names = results.get("bboxes_labels", [])
    num_instances = len(class_names)

    if num_instances == 0:
        print("[Warning] No objects detected by open-vocabulary detection.")
        return {
            "image": image_np,
            "boxes": np.zeros((0, 4)),
            "masks": np.zeros((0, image_np.shape[0], image_np.shape[1]), dtype=bool),
            "class_names": [],
            "class_ids": np.array([], dtype=int),
            "scores": np.array([], dtype=float)
        }

    class_ids = np.arange(num_instances)
    scores = np.ones(num_instances, dtype=np.float32)  # Florence-2 无显式 score，统一置 1.0

    # ---------- 4. SAM2 分割 ----------
    sam2_predictor.set_image(image_np)
    masks, _, _ = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    if masks.ndim == 4:
        masks = masks.squeeze(1)

    # ---------- 5. 可视化（可选） ----------
    if visualize:
        img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        detections = sv.Detections(
            xyxy=input_boxes,
            mask=masks.astype(bool),
            class_id=class_ids,
            confidence=scores
        )

        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        mask_annotator = sv.MaskAnnotator()

        labels = [f"{name}" for name in class_names]

        annotated = box_annotator.annotate(img_bgr.copy(), detections)
        annotated = label_annotator.annotate(annotated, detections, labels)
        masked = mask_annotator.annotate(img_bgr.copy(), detections)

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(
                os.path.join(output_dir, "open_vocab_detection.jpg"),
                annotated
            )
            cv2.imwrite(
                os.path.join(output_dir, "open_vocab_detection_mask.jpg"),
                masked
            )

    # ---------- 6. 返回（关键） ----------
    return {
        "image": image_np,
        "boxes": input_boxes,
        "masks": masks.astype(bool),
        "class_names": class_names,
        "class_ids": class_ids,
        "scores": scores
    }

def open_vocab_per_class_detection(
    florence2_model,
    florence2_processor,
    sam2_predictor,
    image_path,
    class_list,
    output_dir=None,
    visualize=False
):
    """
    逐类调用 open-vocabulary detection
    返回格式与 grounded_sam2_detection 完全一致
    """

    all_boxes = []
    all_masks = []
    all_class_names = []
    all_class_ids = []
    all_scores = []

    image_np = None
    class_id_counter = 0

    for cls_name in class_list:
        print(f"[OpenVocab] Detecting class: {cls_name}")

        results = open_vocabulary_detection_and_segmentation(
            florence2_model=florence2_model,
            florence2_processor=florence2_processor,
            sam2_predictor=sam2_predictor,
            image_path=image_path,
            text_input=cls_name,          # ⭐ 关键：单类
            visualize=visualize,
            output_dir=output_dir
        )

        if results["boxes"].shape[0] == 0:
            continue

        if image_np is None:
            image_np = results["image"]

        n = results["boxes"].shape[0]

        all_boxes.append(results["boxes"])
        all_masks.append(results["masks"])
        all_class_names.extend([cls_name] * n)
        all_class_ids.extend(
            list(range(class_id_counter, class_id_counter + n))
        )
        all_scores.extend(results["scores"])

        class_id_counter += n

    if len(all_boxes) == 0:
        print("[Warning] No objects detected in any class.")
        h, w = image_np.shape[:2]
        return {
            "image": image_np,
            "boxes": np.zeros((0, 4)),
            "masks": np.zeros((0, h, w), dtype=bool),
            "class_names": [],
            "class_ids": np.array([], dtype=int),
            "scores": np.array([], dtype=float)
        }

    return {
        "image": image_np,
        "boxes": np.concatenate(all_boxes, axis=0),
        "masks": np.concatenate(all_masks, axis=0),
        "class_names": all_class_names,
        "class_ids": np.array(all_class_ids),
        "scores": np.array(all_scores)
    }

def merged_open_vocabulary_detection(
    florence2_model,
    florence2_processor,
    sam2_predictor,
    image_path,
    class_list,
    output_dir=None,
    visualize=False
):
    # 1. 使用更加自然语言的 Prompt 格式
    task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
    text_input = "detect " + ", ".join(class_list) # 变为 "detect cup, bottle, umbrella..."
    
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    
    # 2. 运行推理
    results = run_florence2(
        task_prompt, 
        text_input, 
        florence2_model, 
        florence2_processor, 
        image
    )
    
    parsed_results = results.get(task_prompt, {})

    # 3. 检查解析结果
    if "bboxes" not in parsed_results or len(parsed_results["bboxes"]) == 0:
        print(f"[Warning] No objects found for: {text_input}")
        return None

    # 获取 Bboxes 和 Labels
    input_boxes = np.array(parsed_results["bboxes"], dtype=np.float32)
    # 该任务返回 labels 键
    class_names = parsed_results.get("labels", []) 
    
    # 4. 运行 SAM-2 (保持高效：单次 set_image)
    sam2_predictor.set_image(image_np)
    
    masks, scores, _ = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    if masks.ndim == 4:
        masks = masks.squeeze(1)

    return {
        "image": image_np,
        "boxes": input_boxes,
        "masks": masks.astype(bool),
        "class_names": class_names,
        "class_ids": np.arange(len(class_names)),
        "scores": scores
    }



import time
start_time = time.time()

print("开始测试")
elapsed_time = time.time() - start_time
print(f"运行到此步骤耗时: {elapsed_time:.4f} 秒")

print("加载模型")
# 1. 初始化模型
text_encoder_type1 = "/home/frank/workspace/florence2"
FLORENCE2_MODEL_ID = os.path.expanduser(text_encoder_type1)
# FLORENCE2_MODEL_ID = "microsoft/Florence-2-large"
SAM2_CHECKPOINT = "/home/frank/workspace/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# environment settings
# use bfloat16
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# build florence-2
florence2_model = AutoModelForCausalLM.from_pretrained(FLORENCE2_MODEL_ID, trust_remote_code=True, torch_dtype='auto').eval().to(device)
florence2_processor = AutoProcessor.from_pretrained(FLORENCE2_MODEL_ID, trust_remote_code=True)

# build sam 2
sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
sam2_predictor = SAM2ImagePredictor(sam2_model)


elapsed_time = time.time() - start_time
print(f"运行到此步骤耗时: {elapsed_time:.4f} 秒")
print("配置参数")
calib_path = "/home/frank/workspace/Picture_Capture/data/calibration_aruco_final.txt"
# Camera 1 (左) 内参
camera1_intrinsics = np.array([
    [745.9627075195312, 0.0, 638.2297973632812], 
    [0.0, 745.2562866210938, 360.5473937988281],
    [0.0, 0.0, 1.0]
])

# Camera 2 (右) 内参
camera2_intrinsics = np.array([
    [752.1408081054688, 0.0, 642.9800415039062],
    [0.0, 751.44580078125, 360.2501220703125],
    [0., 0., 1.]
])

# 外参 T_calib: 把 camera1 下的点变换到 camera2 坐标系 (cloud2 = T_calib * cloud1)
T_calib = np.linalg.inv(np.loadtxt(calib_path))

# 路径设置
image_path2 = "/home/frank/workspace/Picture_Capture/data/camera2/camera2_color_20260118_132426.png"
image_path1 = image_path2.replace("camera2", "camera1")
depth_path2 = image_path2.replace("color", "depth")
depth_path1 = image_path1.replace("color", "depth")

# 输出目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output_test")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OPEN_VOCAB_CLASSES = ["harmmer", "bottle", "umbrella", "book", "phone", "wrench", "screwdriver"]

# ===========================
# 0) 全局参数（替换原 MAX_DEPTH）
# ===========================
# 深度截断阈值（米）：原来 1.0 会造成大批量丢点，建议先放宽
MAX_DEPTH = 3.0   # 先用 5m；如果你想不截断，改成 None
MIN_POINTS_INSTANCE = 80   # 每个实例最少点数（太小基本是噪声/空mask）
VOXEL_INSTANCE = 0.002
VOXEL_FULL = 0.005

# ==========================================
#    适用于：depth 已 aligned 到 RGB，但分辨率可能不同
# ==========================================
def reconstruct_pcd_from_depth(rgb_img, depth_img, intrinsics_rgb, mask=None):
    """
    从 aligned depth 还原点云：
      - depth 与 rgb 对齐（同一像素语义），但分辨率可不同
      - 内参给的是 RGB 内参，需要缩放到 depth 分辨率再做 back-projection
    """
    rgb_h, rgb_w = rgb_img.shape[:2]
    depth_h, depth_w = depth_img.shape[:2]

    # RGB->Depth 分辨率比例
    scale_x = rgb_w / depth_w
    scale_y = rgb_h / depth_h

    fx, fy = intrinsics_rgb[0, 0], intrinsics_rgb[1, 1]
    cx, cy = intrinsics_rgb[0, 2], intrinsics_rgb[1, 2]

    # 关键：把 RGB 内参缩放到 Depth 像素坐标系
    fx_d = fx / scale_x
    fy_d = fy / scale_y
    cx_d = cx / scale_x
    cy_d = cy / scale_y

    # 构建有效 mask（depth 空值剔除；可选深度上限）
    if mask is not None:
        mask_d = cv2.resize(mask.astype(np.uint8), (depth_w, depth_h), interpolation=cv2.INTER_NEAREST).astype(bool)
        valid = np.logical_and(mask_d, depth_img > 0)
    else:
        valid = depth_img > 0

    if MAX_DEPTH is not None:
        valid = np.logical_and(valid, depth_img < MAX_DEPTH)

    if not np.any(valid):
        return None

    v, u = np.where(valid)
    z = depth_img[v, u]

    # 关键：用缩放后的 cx_d, fx_d 在 depth 像素系里投影
    x = (u - cx_d) * z / fx_d
    y = (v - cy_d) * z / fy_d
    pts = np.stack((x, y, z), axis=-1)

    # 颜色：depth 像素映射回 rgb 像素取色（你原来这部分思路是对的）
    u_rgb = np.clip((u * scale_x).astype(np.int32), 0, rgb_w - 1)
    v_rgb = np.clip((v * scale_y).astype(np.int32), 0, rgb_h - 1)
    cols = rgb_img[v_rgb, u_rgb].astype(np.float32) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    return pcd


def safe_centroid(pcd):
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return None
    return pts.mean(axis=0)


# ==========================================
# 2) 运行流程
# ==========================================

elapsed_time = time.time() - start_time
print(f"[1/7] Running Inference (per-class open-vocab)...运行到此步骤耗时: {elapsed_time:.4f} 秒")
det_res1 = open_vocab_per_class_detection(
    florence2_model=florence2_model,
    florence2_processor=florence2_processor,
    sam2_predictor=sam2_predictor,
    image_path=image_path1,
    class_list=OPEN_VOCAB_CLASSES,
    output_dir=os.path.join(OUTPUT_DIR, "vis_cam1"),
    visualize=False
)
det_res2 = open_vocab_per_class_detection(
    florence2_model=florence2_model,
    florence2_processor=florence2_processor,
    sam2_predictor=sam2_predictor,
    image_path=image_path2,
    class_list=OPEN_VOCAB_CLASSES,
    output_dir=os.path.join(OUTPUT_DIR, "vis_cam2"),
    visualize=False
)

# 保存 mask（调试用）
save_instance_masks(det_res1, os.path.join(OUTPUT_DIR, "instance_masks_1"), prefix="cam1")
save_instance_masks(det_res2, os.path.join(OUTPUT_DIR, "instance_masks_2"), prefix="cam2")

elapsed_time = time.time() - start_time
print(f"[2/7] Loading depth...运行到此步骤耗时: {elapsed_time:.4f} 秒")
d1 = cv2.imread(depth_path1, cv2.IMREAD_UNCHANGED)
d2 = cv2.imread(depth_path2, cv2.IMREAD_UNCHANGED)
assert d1 is not None and d2 is not None, "Failed to load depth images"
d1 = d1.astype(np.float32) / 1000.0
d2 = d2.astype(np.float32) / 1000.0


elapsed_time = time.time() - start_time
print(f"[3/7] Reconstructing instance point clouds...运行到此步骤耗时: {elapsed_time:.4f} 秒")

def get_mean_color_vec(rgb_img, mask):
    """计算余弦相似度用的颜色特征向量（单位向量）"""
    if mask is None or (not np.any(mask)):
        return np.zeros(3, dtype=np.float32)
    pixels = rgb_img[mask].astype(np.float32)  # RGB
    mean_color = np.mean(pixels, axis=0)       # (3,)
    return mean_color / (np.linalg.norm(mean_color) + 1e-6)

pcds1 = []
for m, name in zip(det_res1["masks"], det_res1["class_names"]):
    p = reconstruct_pcd_from_depth(det_res1["image"], d1, camera1_intrinsics, mask=m)
    if p is None:
        continue
    if len(p.points) < MIN_POINTS_INSTANCE:
        continue
    pcds1.append({
        "pcd": p,
        "name": name,
        "feat": get_mean_color_vec(det_res1["image"], m),
        "centroid": safe_centroid(p)
    })

pcds2 = []
for m, name in zip(det_res2["masks"], det_res2["class_names"]):
    p = reconstruct_pcd_from_depth(det_res2["image"], d2, camera2_intrinsics, mask=m)
    if p is None:
        continue
    if len(p.points) < MIN_POINTS_INSTANCE:
        continue
    pcds2.append({
        "pcd": p,
        "name": name,
        "feat": get_mean_color_vec(det_res2["image"], m),
        "centroid": safe_centroid(p)
    })

print(f"  cam1 instances: {len(pcds1)}")
print(f"  cam2 instances: {len(pcds2)}")


# # 1. 生成完整的场景点云用于对齐测试
pcd_full1 = reconstruct_pcd_from_depth(det_res1["image"], d1, camera1_intrinsics, mask=None)
pcd_full2 = reconstruct_pcd_from_depth(det_res2["image"], d2, camera2_intrinsics, mask=None)


# ========= 把 cam1 实例变换到 cam2 坐标系 =========
for o in pcds1:
    o["pcd"].transform(T_calib)
    o["centroid"] = safe_centroid(o["pcd"])

# ========= 实例匹配 + 合并（相同合并，不相同单独保留）=========
elapsed_time = time.time() - start_time
print(f"[5/7] Matching and merging (class + centroid + color)...运行到此步骤耗时: {elapsed_time:.4f} 秒")
MAX_DIST = 0.30        # 18cm（先放宽，确保能匹配；稳定后再收紧）
MIN_COLOR_SIM = 0.75   # 光照差异较大时适当降低
COLOR_W = 0.03

final_objects = []
used1, used2 = set(), set()

all_classes = sorted(set([x["name"] for x in pcds1] + [x["name"] for x in pcds2]))

for cls in all_classes:
    idx1 = [i for i, x in enumerate(pcds1) if x["name"] == cls and x["centroid"] is not None]
    idx2 = [j for j, x in enumerate(pcds2) if x["name"] == cls and x["centroid"] is not None]
    if not idx1 or not idx2:
        continue

    cand = []
    for i in idx1:
        if i in used1:
            continue
        for j in idx2:
            if j in used2:
                continue
            d_xyz = float(np.linalg.norm(pcds1[i]["centroid"] - pcds2[j]["centroid"]))
            sim_c = float(np.dot(pcds1[i]["feat"], pcds2[j]["feat"]))
            if sim_c < MIN_COLOR_SIM:
                continue
            cost = d_xyz + COLOR_W * (1.0 - sim_c)
            cand.append((cost, d_xyz, sim_c, i, j))

    cand.sort(key=lambda t: t[0])
    for cost, d_xyz, sim_c, i, j in cand:
        if i in used1 or j in used2:
            continue
        if d_xyz > MAX_DIST:
            continue

        merged = pcds1[i]["pcd"] + pcds2[j]["pcd"]
        merged = merged.voxel_down_sample(VOXEL_INSTANCE)

        final_objects.append({"pcd": merged, "name": f"merged_{cls}"})
        used1.add(i)
        used2.add(j)
        print(f"  merged {cls}: dist={d_xyz:.3f}m sim={sim_c:.3f}")

# 未匹配的全部保留（保证数量 >= 单相机最大数量）
for i, o in enumerate(pcds1):
    if i not in used1:
        final_objects.append({"pcd": o["pcd"].voxel_down_sample(VOXEL_INSTANCE), "name": f"cam1_only_{o['name']}"})
for j, o in enumerate(pcds2):
    if j not in used2:
        final_objects.append({"pcd": o["pcd"].voxel_down_sample(VOXEL_INSTANCE), "name": f"cam2_only_{o['name']}"})

# ========= 全深度点云合并 =========
elapsed_time = time.time() - start_time
print(f"[6/7] Merging full scene...运行到此步骤耗时: {elapsed_time:.4f} 秒")
if pcd_full1 is not None:
    pcd_full1.transform(T_calib)
merged_full = None
if pcd_full1 is None and pcd_full2 is None:
    merged_full = None
elif pcd_full1 is None:
    merged_full = pcd_full2
elif pcd_full2 is None:
    merged_full = pcd_full1
else:
    merged_full = pcd_full1 + pcd_full2

if merged_full is not None:
    merged_full = merged_full.voxel_down_sample(VOXEL_FULL)

# ========= 保存（与你单相机脚本一致：最后统一 flip）=========
elapsed_time = time.time() - start_time
print(f"[7/7] Saving point clouds...运行到此步骤耗时: {elapsed_time:.4f} 秒")
pcd_seg_dir = os.path.join(OUTPUT_DIR, "pcd_seg")
os.makedirs(pcd_seg_dir, exist_ok=True)

flip_R = o3d.geometry.get_rotation_matrix_from_xyz((np.pi, 0, 0))

for k, obj in enumerate(final_objects):
    p = obj["pcd"]
    if p is None or len(p.points) == 0:
        continue
    
    p.rotate(flip_R, center=(0, 0, 0)) 
    
    o3d.io.write_point_cloud(os.path.join(pcd_seg_dir, f"{obj['name']}_{k}.pcd"), p)
if merged_full is not None and len(merged_full.points) > 0:
    # 这里的场景做了翻转
    merged_full.rotate(flip_R, center=(0, 0, 0))
    o3d.io.write_point_cloud(os.path.join(pcd_seg_dir, "merged_full_scene.pcd"), merged_full)

print(f"[Success] Saved {len(final_objects)} instance clouds (+ full scene) to: {pcd_seg_dir}")
