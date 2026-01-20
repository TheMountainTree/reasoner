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
        image,
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
        image_path: 输入图像
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
    R = pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0))
    pcd.rotate(R, center=(0, 0, 0))

    # 可选: 降采样和去噪
    if len(points) > 1000:
        pcd = pcd.voxel_down_sample(voxel_size=0.005)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    return pcd


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

