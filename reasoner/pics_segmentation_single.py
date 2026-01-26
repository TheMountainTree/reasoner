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
import time
from scipy.spatial import KDTree

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
        # vis.run()
        vis.poll_events()
        vis.update_renderer()

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
    # vis.run()
    vis.poll_events()
    vis.update_renderer()

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


def save_instance_masks(
    detection_results,
    output_dir,
    prefix="mask",
    depth_image=None,
    intrinsics=None,
    save_mask=True,
    save_color=True,
    save_depth=True,
    save_points=True
):
    """
    将 detection_results 中的实例掩码按 类别+编号 保存为文件

    输出示例: {prefix}_{类别名}_{编号}.{后缀}
        person_1.png
        person_1.npy
        person_2.png
        chair_1.png
        table_1.png

    参数:
        detection_results: grounded_sam2_detection 的返回值
        output_dir: 输出目录
        prefix: 文件名前缀
        depth_image: 深度图 (可选)
        intrinsics: 相机内参 (可选, 用于生成点云)
        save_mask: 是否保存掩码
        save_color: 是否保存彩色抠图
        save_depth: 是否保存掩码后的深度图
        save_points: 是否保存掩码后的点云

    输出：
        PNG图片(0-255灰度),方便肉眼查看
        NPY文件:将掩码原始布尔值或数值矩阵保存为 NumPy 格式，用于后续程序进行精确计算（如反向映射坐标）
    """
    os.makedirs(output_dir, exist_ok=True)

    masks = detection_results["masks"]
    class_names = detection_results["class_names"]

    image_rgb = detection_results["image"]
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

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
        
        saved_items = []

        # ---------- 保存为 PNG/NPY（Mask） ----------
        if save_mask:
            mask_img = (mask.astype(np.uint8) * 255)
            png_path = os.path.join(output_dir, f"{prefix}_{base_name}.png")
            cv2.imwrite(png_path, mask_img)

            npy_path = os.path.join(output_dir, f"{prefix}_{base_name}.npy")
            np.save(npy_path, mask)
            saved_items.append(f"Mask->{png_path}")

        # ---------- 保存为 RGB Cutout (彩色图) ----------
        if save_color:
            # 创建一个纯黑的背景
            color_cutout = np.zeros_like(image_bgr)
            # 将原图中mask为true的部分复制到纯黑背景上
            color_cutout[mask] = image_bgr[mask]

            color_path = os.path.join(output_dir, f"{prefix}_{base_name}_color.png")
            cv2.imwrite(color_path, color_cutout)
            saved_items.append(f"Color->{color_path}")
        
        # ---------- 处理深度相关 (Depth/Points) ----------
        if depth_image is not None:
            # Resize mask to match depth resolution
            dh, dw = depth_image.shape[:2]
            # We resize mask to depth size using nearest neighbor
            mask_resized = cv2.resize(mask.astype(np.uint8), (dw, dh), interpolation=cv2.INTER_NEAREST).astype(bool)

            # 1. 保存 Masked Depth
            if save_depth:
                # Create masked depth
                masked_depth = np.zeros_like(depth_image)
                masked_depth[mask_resized] = depth_image[mask_resized]
                
                # Save NPY (Float, Meters)
                npy_depth_path = os.path.join(output_dir, f"{prefix}_{base_name}_depth.npy")
                np.save(npy_depth_path, masked_depth)
                saved_items.append(f"Depth->{npy_depth_path}")

            # 2. 保存 Masked Points (点云) - Organized (H, W, 3)
            if save_points and intrinsics is not None:
                height, width = depth_image.shape
                fx, fy = intrinsics[0, 0], intrinsics[1, 1]
                cx, cy = intrinsics[0, 2], intrinsics[1, 2]
                
                # 生成网格坐标
                u, v = np.meshgrid(np.arange(width), np.arange(height))
                
                # 准备 Z (Masked)
                z = np.zeros_like(depth_image)
                # 仅在 mask 区域且深度有效时保留深度值
                valid_mask = np.logical_and(mask_resized, depth_image > 0)
                z[valid_mask] = depth_image[valid_mask]
                
                # 计算 X, Y
                # 注意：当 z=0 时，x, y 也会是 0 (或 NaN/Inf，取决于具体计算，这里 z/fx 是 0，所以 (u-cx)*0 = 0)
                # 但为了安全，我们可以利用 numpy 的广播机制直接计算，z 为 0 处结果即为 0
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                
                # 组合为 (H, W, 3)
                points_map = np.stack((x, y, z), axis=-1)
                
                npy_points_path = os.path.join(output_dir, f"{prefix}_{base_name}_points.npy")
                np.save(npy_points_path, points_map)
                saved_items.append(f"Points->{npy_points_path}")

        print(f"[Saved] {base_name}: " + ", ".join(saved_items))


# Open-Vocabulary Detection + Segmentation
def open_vocabulary_detection_and_segmentation(
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
    # text_input = "detect " + ", ".join(class_list) # 变为 "detect cup, bottle, umbrella..."
    text_input = class_list
    
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

    if scores.ndim > 1: scores = scores.flatten()

    if visualize and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        for i in range(len(input_boxes)):
            # 1. 提取单个检测对象的数据 (保持维度)
            box = input_boxes[i:i+1]
            mask = masks[i:i+1]
            class_id = np.array([i]) # 或者使用 class_ids[i:i+1]
            score = scores[i:i+1] if scores is not None else np.array([1.0])
            label = class_names[i]

            # 2. 构建单个对象的 Detections
            single_detection = sv.Detections(
                xyxy=box,
                mask=mask.astype(bool),
                class_id=class_id,
                confidence=score
            )

            # 3. 绘制标注 (基于原始图像的拷贝)
            annotated_frame = img_bgr.copy()

            # Box
            box_annotator = sv.BoxAnnotator()
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=single_detection)

            # Label
            label_annotator = sv.LabelAnnotator()
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=single_detection, labels=[label]
            )

            # Mask
            mask_annotator = sv.MaskAnnotator()
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=single_detection)

            # 4. 保存单张图片
            save_name = f"detection_{i}_{label}.jpg"
            cv2.imwrite(os.path.join(output_dir, save_name), annotated_frame)
            print(f"Saved visualization: {save_name}")

    det_res = {
        "image": image_np,
        "boxes": input_boxes,
        "masks": masks.astype(bool),
        "class_names": class_names,
        "class_ids": np.arange(len(class_names)),
        "scores": scores
    }

    det_res = dedup_det_res_by_mask_iou(det_res, iou_thr=0.90)

    return det_res

    # return {
    #     "image": image_np,
    #     "boxes": input_boxes,
    #     "masks": masks.astype(bool),
    #     "class_names": class_names,
    #     "class_ids": np.arange(len(class_names)),
    #     "scores": scores
    # }


def get_fitness(pcd1, pcd2, max_dist=0.01):
    """
    计算两个点云的重合度 (Fitness)
    target -> source 双向最大覆盖率
    np.eye(4)意味着点云不会发生变换，直接进行比对计算
    最终返回最大的重合度分数
        1.0 source cloud 与 target cloud 完全重合
        0.0 完全不重合或者为空
    """
    if len(pcd1.points) == 0 or len(pcd2.points) == 0:
        return 0.0
    # source -> target
    res1 = o3d.pipelines.registration.evaluate_registration(
        pcd1, pcd2, max_dist, np.eye(4)
    )
    # target -> source
    res2 = o3d.pipelines.registration.evaluate_registration(
        pcd2, pcd1, max_dist, np.eye(4)
    )
    return max(res1.fitness, res2.fitness)

def get_obb_distance(pcd1, pcd2):
    """
    计算两个点云 OBB (Oriented Bounding Box, 有向包围盒) 的极其近似的'表面距离'
    Oriented Bounding Box(OBB,有向包围盒) 是一种用于包裹或包围物体的矩形(2D)或长方体(3D)边界框,
    与传统的轴对齐包围盒(AABB, Axis-Aligned Bounding Box)不同,OBB 可以根据物体的方向进行旋转，使其与物体的主方向对齐
    OBB相比AABB可以保证紧凑性与准确性(主轴对齐),旋转不变性(BB会随着物体旋转而动,保证几何关系不变)
    Dist = |C1 - C2| - (Extent1/2 + Extent2/2)
    估算半径extent将包围盒近似为一个球体
    如果结果 < 0,说明包围盒相交或包含
    """
    try:
        obb1 = pcd1.get_oriented_bounding_box()
        obb2 = pcd2.get_oriented_bounding_box()
        
        # 两个中心点的欧氏距离
        center_dist = np.linalg.norm(obb1.center - obb2.center)
        
        # 估算半径 (取长宽高中最大的那一半，作为包围球半径的保守估计，或者对角线的一半)
        # 这里为了保守起见，用最大轴长的一半 (Longest dim radius)
        # 如果要更宽松，可以用 np.linalg.norm(obb.extent) / 2.0 (对角线一半)
        r1 = np.linalg.norm(obb1.extent) / 2.0
        r2 = np.linalg.norm(obb2.extent) / 2.0
        
        # 表面距离近似值
        surface_dist = center_dist - (r1 + r2)
        return surface_dist
    except:
        return 999.0

def merge_pcd_list(obj_list, voxel_size):
    """
    将一组包含点云的对象字典(list of objects)合并为一个单一的对象字典
    合并一个列表中的所有对象点云，并更新中心和体积
    作用是为Graph进行节点更新,合并被视为同个物体的点云

    参数说明:
        obj_list: 每个元素都是一个字典，包含'pcd'(Open3D PointCloud对象)和'name'等键
        voxel_size: 浮点数,用于对合并后的点云进行体素降采样的尺寸参数

    过程：
        空检查: 如果obj_list为空直接返回None
        初始化: 
            创建一个空的Open3D点云对象merged_pcd
            提取列表中第一个对象的'name'在以巍峨u合并后对+象的名称
        点云拼接：
            遍历列表中的所有对象，使用 += 运算符将它们的点云数据(obj['pcd'])累加到 merged_pcd 中
        统一降采样:
            merged_pcd.voxel_down_sample(voxel_size): 对拼接后的整体点云进行体素下采样。这有助于减少点云数量、去除重叠区域的冗余点，并使点云密度均匀化
        重新计算属性:
            质心(centroid): 计算合并后云的几何中心
            体积(volume): 尝试计算点云OBB的体积
        构造返回值:
            返回一个新字典,包含以下内容
                pcd: 合并并降采样后的点云对象
                name: 沿用的对象名称
                centroid: 新的质心坐标
                volume: 新的体积
                camera_source: 标记对象来源, merge表示是多个对象合并而来的
    """
    if not obj_list:
        return None
    
    merged_pcd = o3d.geometry.PointCloud()
    name = obj_list[0]['name'] # 沿用第一个的名字
    
    for obj in obj_list:
        merged_pcd += obj['pcd']
    
    # 合并后统一下采样
    merged_pcd = merged_pcd.voxel_down_sample(voxel_size)
    
    # 重新计算属性
    centroid = merged_pcd.get_center() if len(merged_pcd.points) > 0 else np.zeros(3)
    try:
        obb = merged_pcd.get_oriented_bounding_box()
        volume = obb.volume()
    except:
        volume = 0.0
        
    return {
        "pcd": merged_pcd,
        "name": name,
        "centroid": centroid,
        "volume": volume,
        "camera_source": "merged" 
    }

def find_connected_components(nodes, edges):
    """
    连通分量寻找算法
    在给定的图(graph)在中寻找所有连通的分量
    输入:
        nodes: 所有节点的列表或集合
        edges: 边的列表,通常是元组(u, v)的形式,表和四节点u和v相连
    输出:
        components: 一个包含多个集合(set)的列表。每个集合代表一个连通分量，包含该分量内的所有节点
    """
    # 构建邻接表(Adjacency List)
    # 初始化一个字典adj,为每个节点准备一个空集合
    # 遍历输入的egdes,构建无向图。注意这里是双向的
    adj = {i: set() for i in nodes}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # 初始化遍历状态
    # visited: 用于记录归类到某个连通分量中的节点，防止重复处理
    # components: 最终用于存储结果的列表
    visited = set()
    components = []
    
    # 遍历节点并搜索
    # 遍历每一个节点 i
    for i in nodes:
        # 节点没被访问
        if i not in visited:
            # 把所有通过当前节点 i 能连到的节点加入到component集合中
            component = set() # 创建一个空集合，用于存储当前连通分量中的所有节点
            stack = [i] # 初始化栈，放入起始节点 i
            visited.add(i) # 标记起始节点 i 为已访问，防止重复处理
            while stack:
                node = stack.pop() # 从栈中取出一个节点
                component.add(node) # 将其加入当前的组件集合中
                for neighbor in adj[node]: # adj 是邻接表，adj[node] 存储了与 node 相连的所有邻居
                    if neighbor not in visited:
                        visited.add(neighbor) # 标记邻居为已访问
                        stack.append(neighbor) # 将邻居加入栈中，以便下一轮循环继续探索它的邻居
            components.append(component)
    return components

def clustering_objects(object_list, 
                       dist_thr, 
                       vol_sim_thr, 
                       fitness_thr, 
                       obb_dist_thr=0.05,
                       use_volume=True, 
                       min_size_ratio=0.15):
    """
    核心聚类函数:
    对3D点云物体片段进行后处理聚类(合并)
    Logic: SameClass AND (SizeRatio OK) AND ((Dist<T) OR (OBB_Dist<T) OR (Fitness>T) OR (VolSim>T))
    输入
        object_list: 输入物体片段列表(包括点云、名称、质心等信息)
        dist_thr: 质心距离阈值
        vol_sim_thr: 体积相似度阈值
        fitness_thr: 形状相似度阈值
        obb_dist_thr: OBB边缘距离阈值
        min_size_ratio: 最小尺寸比例阈值
    """
    # 如果输入列表为空，直接返回空列表
    if not object_list:
        return []

    # 构建graph，将每个物体视为一个节点(Node)
    n = len(object_list)
    nodes = list(range(n))
    edges = []
    
    # 遍历每一对物体(i和j)，计算两两之间的关系
    for i in range(n):
        for j in range(i + 1, n):
            obj1 = object_list[i]
            obj2 = object_list[j]
            name1, name2 = obj1['name'], obj2['name']
            
            # --- 1. 类别硬过滤 ---
            if name1 != name2:
                continue
            
            # --- 2. 尺寸比例过滤 (防止小噪点粘连大物体) ---
            s1 = len(obj1['pcd'].points)
            s2 = len(obj2['pcd'].points)
            size_ratio = min(s1, s2) / (max(s1, s2) + 1e-9)
            
            if size_ratio < min_size_ratio:
                continue

            # --- 3. 几何特征计算 ---
            # A. 质心距离
            dist = np.linalg.norm(obj1['centroid'] - obj2['centroid'])
            
            # B. OBB 距离 (处理长条物体)
            obb_d = get_obb_distance(obj1['pcd'], obj2['pcd'])
            
            # C. Fitness
            fitness = 0.0
            if obb_d < 0.2: 
                fitness = get_fitness(obj1['pcd'], obj2['pcd'])
            
            # D. 体积相似度
            vol_pass = False
            if use_volume:
                v1, v2 = obj1['volume'], obj2['volume']
                vol_sim = min(v1, v2) / (max(v1, v2) + 1e-9)
                if vol_sim > vol_sim_thr:
                    vol_pass = True

            # --- 4. 决策逻辑 (OR Logic) ---
            is_connected = False
            
            if dist < dist_thr:
                is_connected = True
                # print(f"    Edge {i}-{j} ({name1}): Centroid Dist Pass ({dist:.3f})")
            elif (obb_d < obb_dist_thr) and (dist < dist_thr * 0.8): # obb_d < obb_dist_thr
                is_connected = True
                # print(f"    Edge {i}-{j} ({name1}): OBB Dist Pass ({obb_d:.3f})")
            elif (fitness > fitness_thr) and (dist < dist_thr * 0.8): # fitness > fitness_thr
                is_connected = True
                # print(f"    Edge {i}-{j} ({name1}): Fitness Pass ({fitness:.2f})")
            elif vol_pass and (dist < dist_thr * 2.0):
                is_connected = True
                # print(f"    Edge {i}-{j} ({name1}): Volume Pass ({vol_sim:.2f})")
                
            if is_connected:
                edges.append((i, j)) # 判断连接则添加一个边
                
    # --- 5. 连通分量合并 ---
    components = find_connected_components(nodes, edges) # 找到所有连通子图（例如：A连B，B连C，那么A、B、C就是一组）
    
    merged_results = []
    for comp_ids in components:
        comp_objs = [object_list[idx] for idx in comp_ids]
        merged_obj = merge_pcd_list(comp_objs, voxel_size=0.002)
        if merged_obj:
            merged_results.append(merged_obj)
            
    return merged_results

# ==========================================
#    相机使用depth_registration=true参数来保证depth和color分辨率、光轴和坐标系对齐
# ==========================================
def reconstruct_pcd_from_depth(
        rgb_img, 
        depth_img, 
        intrinsics_rgb, 
        mask=None,
        T_depth2color=None
        ):
    """
    从 aligned depth 还原点云：
      - depth 与 rgb 对齐（同一像素语义），但分辨率可不同
      - 内参给的是 RGB 内参，需要缩放到 depth 分辨率再做 back-projection
    """
    rgb_h, rgb_w = rgb_img.shape[:2]
    depth_h, depth_w = depth_img.shape[:2]

    # RGB->Depth 分辨率比例
    # scale_x = rgb_w / depth_w
    # scale_y = rgb_h / depth_h
    scale_x = 1
    scale_y = 1

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

    # === 新增：depth → color optical frame ===
    if T_depth2color is not None:
        pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])  # Nx4
        pts = (T_depth2color @ pts_h.T).T[:, :3]


    # 颜色：depth 像素映射回 rgb 像素取色（你原来这部分思路是对的）
    u_rgb = np.clip((u * scale_x).astype(np.int32), 0, rgb_w - 1)
    v_rgb = np.clip((v * scale_y).astype(np.int32), 0, rgb_h - 1)
    cols = rgb_img[v_rgb, u_rgb].astype(np.float32) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    return pcd


def safe_centroid(pcd):
    """
    安全地计算点云质心。
    将点云数据转换为numpy数组后计算其几何中心(均值)。
    如果点云为空,则返回None以避免运行时错误。
    Args:
        pcd (open3d.geometry.PointCloud): 输入的Open3D点云对象。
    Returns:
        np.ndarray or None: 若点云非空,返回形状为(3,)的质心坐标数组[x, y, z];
                            若点云为空,返回None。
    """
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return None
    return pts.mean(axis=0)

def mad_filter_pcd_z(pcd, k=3.5, z_min=0.1, z_max=3.0, keep_colors=True):
    """
    对 open3d.geometry.PointCloud 做 z 维度的 median/MAD 去离群。
    - 先做有限性/深度范围过滤，再做 MAD。
    - k: 阈值倍数，默认 3.5（常用折中）
    """
    if pcd is None or len(pcd.points) == 0:
        return pcd

    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors) if keep_colors and pcd.has_colors() else None

    # 1) 有效性 + 深度范围过滤
    valid = np.isfinite(pts).all(axis=1)
    valid &= (pts[:, 2] > z_min) & (pts[:, 2] < z_max)

    pts = pts[valid]
    if cols is not None:
        cols = cols[valid]

    if pts.shape[0] == 0:
        return o3d.geometry.PointCloud()

    # 2) median/MAD
    z = pts[:, 2]
    z_med = np.median(z)
    mad = np.median(np.abs(z - z_med)) + 1e-9
    sigma = 1.4826 * mad
    keep = np.abs(z - z_med) < (k * sigma)

    pts = pts[keep]
    if cols is not None:
        cols = cols[keep]

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts)
    if cols is not None:
        out.colors = o3d.utility.Vector3dVector(cols)
    return out


def build_object_list(det_res, depth_img, intrinsics, T_depth2color=None, transform=None, suffix=""):
    """
    根据 2D 检测结果和深度图构建 3D 物体列表。
    该函数遍历检测结果中的每一个掩膜，将其反投影为 3D 点云，并进行过滤、
    变换（如多相机对齐）、下采样以及基础几何属性（质心、体积）的计算。
    Args:
        det_res (dict): 目标检测结果字典，需包含以下键:
            - "masks": 分割掩膜列表 (List[np.array])
            - "class_names": 对应的类别名称列表 (List[str])
            - "image": 原图 (用于纹理映射，视 reconstruct_pcd_from_depth 实现而定)
        depth_img (np.array): 对应的深度图像。
        intrinsics (o3d.camera.PinholeCameraIntrinsic 或 np.array): 相机内参矩阵。
        T_depth2color (np.array, optional): 深度相机到彩色相机的变换矩阵。默认为 None。
        transform (np.array, optional): 额外的刚体变换矩阵 (4x4)。
            通常用于将当前相机的坐标系变换到主相机或世界坐标系 (例如 Cam1 -> Cam2)。默认为 None。
        suffix (str, optional): 生成物体 ID 时使用的后缀字符串。默认为 ""。
    Returns:
        list[dict]: 处理后的物体对象列表。每个元素为一个字典，包含以下键:
            - "pcd" (o3d.geometry.PointCloud): 处理后的 Open3D 点云对象。
            - "name" (str): 物体类别名称。
            - "centroid" (np.array): 物体 3D 质心坐标 [x, y, z]。
            - "volume" (float): 物体的定向包围盒 (OBB) 体积。如果计算失败则为 0.0。
            - "id_raw" (str): 原始 ID 标识，格式为 "{name}_{suffix}"。
    Global Constants Used:
        - MIN_POINTS_INSTANCE (int): 判定为有效物体的最小点数阈值。
        - VOXEL_INSTANCE (float): 用于点云下采样的体素大小。
    """
    obj_list = []
    # 重建
    for m, name in zip(det_res["masks"], det_res["class_names"]):
        p = reconstruct_pcd_from_depth(det_res["image"], depth_img, intrinsics, mask=m, T_depth2color=T_depth2color)
        
        # 基础过滤
        if p is None or len(p.points) < MIN_POINTS_INSTANCE:
            continue
            
        # 坐标变换 (如果是Cam1，变换到Cam2坐标系)
        if transform is not None:
            p.transform(transform)
        
        # 预处理：下采样
        p = p.voxel_down_sample(VOXEL_INSTANCE)
        
        # median/MAD 鲁棒去离群
        p = mad_filter_pcd_z(
            p,
            k=3.5,                 # 可调：3.0更严格 / 4.0更宽松
            z_min=0.1,
            z_max=MAX_DEPTH if MAX_DEPTH is not None else 3.0
        )

        # 过滤后再做一次点数检查
        if p is None or len(p.points) < MIN_POINTS_INSTANCE:
            continue

        # 计算基础属性
        centroid = p.get_center()
        try:
            obb = p.get_oriented_bounding_box()
            volume = obb.volume()
        except:
            volume = 0.0
            
        obj_list.append({
            "pcd": p,
            "name": name,
            "centroid": centroid,
            "volume": volume,
            "id_raw": f"{name}_{suffix}" 
        })
    return obj_list

def mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """mask1/mask2: bool(H,W)"""
    m1 = mask1.astype(bool)
    m2 = mask2.astype(bool)
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum() + 1e-9
    return float(inter) / float(union)

def dedup_det_res_by_mask_iou(det_res, iou_thr=0.90, prefer="score_then_area"):
    """
    对 merged_open_vocabulary_detection 的输出做去重（跨类别也去重）
    - iou_thr: 两个 mask IoU 超过该阈值，视为同一实例的重复识别
    - prefer: 保留策略
        - "score_then_area": score 大的优先；score 相同保留 mask 面积大的
        - "area": 只按 mask 面积
    返回：同结构 det_res（boxes/masks/class_names/scores 都同步裁剪）
    """
    if det_res is None:
        return None

    masks = det_res["masks"]
    boxes = det_res["boxes"]
    names = det_res["class_names"]
    scores = det_res.get("scores", None)

    n = len(names)
    if n <= 1:
        return det_res

    keep = np.ones(n, dtype=bool)

    # 简单 NMS：按优先级排序后，保留一个，压掉与其 IoU 高的其他项
    if scores is not None and prefer.startswith("score"):
        order = np.argsort(-scores)  # 高分在前
    else:
        areas = np.array([m.sum() for m in masks], dtype=np.float32)
        order = np.argsort(-areas)

    areas = np.array([m.sum() for m in masks], dtype=np.float32)

    for _i in range(n):
        i = int(order[_i])
        if not keep[i]:
            continue
        for _j in range(_i + 1, n):
            j = int(order[_j])
            if not keep[j]:
                continue

            iou = mask_iou(masks[i], masks[j])
            if iou >= iou_thr:
                # 决定删谁（默认删 j，因为 i 在排序里优先级更高）
                keep[j] = False

    idx = np.where(keep)[0]

    return {
        "image": det_res["image"],
        "boxes": boxes[idx],
        "masks": masks[idx],
        "class_names": [names[k] for k in idx],
        "class_ids": np.arange(len(idx), dtype=int),
        "scores": (scores[idx] if scores is not None else np.ones(len(idx), dtype=np.float32)),
    }


if __name__ == "__main__":
    start_time = time.time()
    total_start = time.time()

    print("开始测试 (Single Camera - Cam1)")
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

    # 相机内参配置
    elapsed_time = time.time() - start_time
    print(f"运行到此步骤耗时: {elapsed_time:.4f} 秒")
    print("配置参数")
    
    # Camera 1 (左) 内参
    camera1_intrinsics = np.array([
        [745.9627075195312, 0.0, 638.2297973632812], 
        [0.0, 745.2562866210938, 360.5473937988281],
        [0.0, 0.0, 1.0]
    ])

    T_depth2color_cam1 = np.array([
        [ 1.000, -0.004, -0.004,  0.032],
        [ 0.004,  0.994, -0.105,  0.001],
        [ 0.004,  0.105,  0.994, -0.002],
        [ 0.000,  0.000,  0.000,  1.000]
    ])

    # 路径设置 (使用 Camera 1)
    image_path1 = "/home/frank/workspace/Picture_Capture/data/camera1/camera1_color_20260122_144954.png"
    depth_path1 = image_path1.replace("color", "depth")

    # 输出目录
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output_single")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # OPEN_VOCAB_CLASSES = ["harmmer", "bottle", "phone", "book", "wrench", "screwdriver"]
    OPEN_VOCAB_CLASSES = "harmmer, bottle, phone, book, wrench, screwdriver"

    # ===========================
    # 0) 全局参数（替换原 MAX_DEPTH）
    # ===========================
    MAX_DEPTH = 3.0   # 深度截断阈值（米）；如果你想不截断，改成 None
    MIN_POINTS_INSTANCE = 800   # 每个实例最少点数（太小基本是噪声/空mask）
    VOXEL_INSTANCE = 0.002
    VOXEL_FULL = 0.005

    # ==========================================
    # 2) 运行流程
    # ==========================================
    t_step_start = time.time()
    print(f"\n====== [1/6] Loading Depth Image ======")
    # 注意：深度图单位一般是毫米，除此之外还需要除以1000变为米
    d1 = cv2.imread(depth_path1, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    print(f"-- Depth Load Time: {time.time() - t_step_start:.4f}s")

    t_step_start = time.time()
    print(f"\n====== [2/6] Running Inference (Florence2 + SAM2) - Camera 1 ======")
    
    # 基于prompt的此表一次查询
    det_res1 = merged_open_vocabulary_detection(
        florence2_model, florence2_processor, sam2_predictor, image_path1, OPEN_VOCAB_CLASSES,
        visualize=True, output_dir=os.path.join(OUTPUT_DIR, "vis_cam1")
    )

    
    save_instance_masks(
        det_res1, 
        os.path.join(OUTPUT_DIR, "instance_masks_1"), 
        prefix="cam1", 
        depth_image=d1,
        intrinsics=camera1_intrinsics
    )
    print(f"-- Inference Time: {time.time() - t_step_start:.4f}s")
    
    t_step_start = time.time()
    print(f"\n====== [3/6] Reconstructing Instance Point Clouds ======")
    # 原始实例列表 (Camera 1 坐标系, 不进行 T_calib 变换)
    raw_objs1 = build_object_list(det_res1, d1, camera1_intrinsics, T_depth2color=T_depth2color_cam1, transform=None, suffix="c1")
    print(f"-- Raw Instances: Cam1={len(raw_objs1)}")
    print(f"-- Reconstruction Time: {time.time() - t_step_start:.4f}s")
    
    t_step_start = time.time()

    print(f"\n====== [4/6] Step 1: Intra-Camera Merge (Self-Correction) ======")
    # 策略：不看 Volume (use_volume=False)，严苛距离，适中 Fitness
    INTRA_DIST_THR = 0.015      # 8cm 单相机内质心距离阈值
    INTRA_OBB_THR = 0.003       # 2cm OBB 间距 (几乎接触)
    INTRA_FITNESS_THR = 0.60   # 有一定重合

    clean_objs1 = clustering_objects(raw_objs1, 
                                        dist_thr=INTRA_DIST_THR, 
                                        vol_sim_thr=0.0, 
                                        fitness_thr=INTRA_FITNESS_THR,
                                        obb_dist_thr=INTRA_OBB_THR,
                                        use_volume=False, # 关掉 Volume
                                        min_size_ratio=0.15)

    # 防御性确保不为None
    clean_objs1 = clean_objs1 if clean_objs1 is not None else []

    print(f"-- After Intra-Merge: Cam1={len(clean_objs1)}")
    print(f"-- Intra-Merge Time: {time.time() - t_step_start:.4f}s")

    # 单视角下，Intra-Merge 的结果即为最终结果，或者如果需要可以再跑一次 relaxed merge (Step 2 logic)
    # 但 usually Step 1 is sufficient for single view to merge over-segmented parts.
    final_objects = clean_objs1
    
    # 打印最终结果
    print(f"--------------------------------------------------")
    print(f"Final Objects Detected: {len(final_objects)}")
    for i, obj in enumerate(final_objects):
        print(f"  [{i}] {obj['name']} | Pts: {len(obj['pcd'].points)} | Vol: {obj['volume']:.4f}")
    print(f"--------------------------------------------------")
    
    t_step_start = time.time()
    print(f"\n====== [5/6] Reconstructing Full Scene Point Clouds ======")
    # 单独构建 Camera 1 全景
    pcd_full1 = reconstruct_pcd_from_depth(det_res1["image"], d1, camera1_intrinsics, mask=None, T_depth2color=T_depth2color_cam1)
    
    # 下采样
    if pcd_full1 is not None: pcd_full1 = pcd_full1.voxel_down_sample(VOXEL_FULL)

    print(f"-- Full Scene Reconstruction Time: {time.time() - t_step_start:.4f}s")
    t_step_start = time.time()
    print(f"\n====== [6/6] Saving Results ======")
    pcd_seg_dir = os.path.join(OUTPUT_DIR, "pcd_seg")
    os.makedirs(pcd_seg_dir, exist_ok=True)

    flip_R = o3d.geometry.get_rotation_matrix_from_xyz((np.pi, 0, 0)) # 翻转以适应Open3D Viewer
    
    # 1. 保存分割出的物体
    for k, obj in enumerate(final_objects):
        p = obj["pcd"]
        if p is None or len(p.points) == 0:
            continue
        p_to_save =  o3d.geometry.PointCloud(p) #copy
        # p_to_save.rotate(flip_R, center=(0, 0, 0)) 
        
        filename = f"{obj['name']}_{k}.pcd"
        o3d.io.write_point_cloud(os.path.join(pcd_seg_dir, filename), p_to_save)

    # 2. 保存全景
    if pcd_full1 is not None and len(pcd_full1.points) > 0:
        p1_save = o3d.geometry.PointCloud(pcd_full1)
        # p1_save.rotate(flip_R, center=(0, 0, 0))
        o3d.io.write_point_cloud(os.path.join(pcd_seg_dir, "cam1_full_scene_aligned.pcd"), p1_save)
        
    print(f"-- Saving Time: {time.time() - t_step_start:.4f}s")
    print(f"\n====== All Completed in {time.time() - total_start:.4f}s ======")
    print(f"[Success] Saved to: {pcd_seg_dir}")
