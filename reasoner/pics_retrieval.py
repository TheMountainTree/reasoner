import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import os
import cv2
import time

class ImagePublisherNode(Node):
    def __init__(self):
        super().__init__('image_publisher_node')
        self.bridge = CvBridge()
        self.processing_enabled = False
        self.images_published = False  # 用于确保只发布一次
        self.image_dir = os.path.join(os.getcwd(), 'segmented_objects')  # 假设图片存储在此目录下

        # 订阅状态主题
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

        # 创建图像发布者
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.img_pub = self.create_publisher(Image, '/image_seg', qos_profile=image_qos)

    def state_callback(self, msg: String):
        if msg.data.strip() == 'E' and not self.images_published:
            self.processing_enabled = True
            self.publish_images()

    def publish_images(self):
        if not self.processing_enabled or self.images_published:
            return
        
        try:
            for img_name in os.listdir(self.image_dir):
                if img_name.endswith('.jpg'):  # 假设仅处理png格式的图片
                    img_path = os.path.join(self.image_dir, img_name)
                    cv_image = cv2.imread(img_path)
                    
                    ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
                    self.img_pub.publish(ros_image)
                    self.get_logger().info(f"Published {img_name} as ROS Image")
                    
                    # 模拟间隔，避免过快发送
                    time.sleep(0.2)  # 根据需要调整延迟时间
            
            self.images_published = True
            self.get_logger().info("Finished publishing all images.")
            
        except Exception as e:
            self.get_logger().error(f"Error processing and publishing images: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()