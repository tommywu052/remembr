"""
CaptionerNode for wheeled-legged robot.
Uses Remote Ollama Gemma4:e4b for image captioning instead of local VILA/NanoLLM.
Subscribes to RealSense image topic, publishes captions to /caption.
"""

import os
import time
import io
from threading import Thread

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

import numpy as np

try:
    from cv_bridge import CvBridge
    HAS_CV_BRIDGE = True
except ImportError:
    HAS_CV_BRIDGE = False
    print("[CaptionerNode] cv_bridge not available, will use raw conversion")

from ollama_client import OllamaClient


class CaptionerNode(Node):

    def __init__(self):
        super().__init__("CaptionerNode")

        self.declare_parameter("ollama_host", "192.168.31.63")
        self.declare_parameter("ollama_port", 11434)
        self.declare_parameter("vlm_model", "gemma4:e4b")
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("caption_topic", "/caption")
        self.declare_parameter("caption_interval", 5.0)
        self.declare_parameter("use_every_nth_image", 15)
        self.declare_parameter("caption_image_count", 1)
        self.declare_parameter("save_images", True)
        self.declare_parameter("image_save_dir", "./memory_images")
        self.declare_parameter(
            "prompt",
            "Please describe in detail what you see in this image. "
            "Focus on people, objects, environmental features, activities, "
            "and spatial layout. Be specific and concise."
        )

        self.ollama = OllamaClient(
            host=self.get_parameter("ollama_host").value,
            port=self.get_parameter("ollama_port").value,
        )
        self.vlm_model = self.get_parameter("vlm_model").value
        self.prompt = self.get_parameter("prompt").value
        self.caption_interval = self.get_parameter("caption_interval").value
        self.use_every_nth = self.get_parameter("use_every_nth_image").value
        self.caption_image_count = self.get_parameter("caption_image_count").value

        if HAS_CV_BRIDGE:
            self.cv_bridge = CvBridge()

        self.image_subscriber = self.create_subscription(
            Image,
            self.get_parameter("image_topic").value,
            self.image_callback,
            10,
        )
        self.caption_publisher = self.create_publisher(
            String,
            self.get_parameter("caption_topic").value,
            10,
        )

        self.save_images = self.get_parameter("save_images").value
        self.image_save_dir = self.get_parameter("image_save_dir").value
        if self.save_images:
            os.makedirs(self.image_save_dir, exist_ok=True)

        self.image_buffer = []
        self.image_counter = 0
        self.caption_loop_running = False
        self.caption_loop_thread = None
        self.logger = self.get_logger()

    def start_caption_loop(self):
        self.caption_loop_running = True
        self.caption_loop_thread = Thread(target=self._caption_loop, daemon=True)
        self.caption_loop_thread.start()
        self.logger.info("Caption loop started")

    def stop_caption_loop(self):
        self.caption_loop_running = False
        if self.caption_loop_thread:
            self.caption_loop_thread.join(timeout=10)
        self.logger.info("Caption loop stopped")

    def image_callback(self, msg: Image):
        self.image_counter += 1
        if self.image_counter % self.use_every_nth != 0:
            return

        try:
            if HAS_CV_BRIDGE:
                cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "rgb8")
            else:
                h, w = msg.height, msg.width
                if msg.encoding in ("rgb8", "bgr8"):
                    cv_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
                    if msg.encoding == "bgr8":
                        cv_image = cv_image[:, :, ::-1]
                else:
                    self.logger.warn(f"Unsupported encoding: {msg.encoding}")
                    return

            self.image_buffer.append(cv_image)
            if len(self.image_buffer) > self.caption_image_count:
                self.image_buffer = self.image_buffer[-self.caption_image_count:]
        except Exception as e:
            self.logger.error(f"Failed to process image: {e}")

    def _caption_loop(self):
        last_caption_time = time.perf_counter()
        self.logger.info(
            f"Waiting for images on {self.get_parameter('image_topic').value} "
            f"(every {self.use_every_nth}th frame, interval={self.caption_interval}s)"
        )

        while self.caption_loop_running:
            time.sleep(self.caption_interval)

            if not self.image_buffer:
                self.logger.info("No images in buffer yet, waiting...")
                continue

            image = self.image_buffer[-1].copy()
            try:
                import cv2
                _, jpeg_bytes = cv2.imencode(".jpg", image[:, :, ::-1])
                jpeg_bytes = jpeg_bytes.tobytes()
            except ImportError:
                from PIL import Image as PILImage
                pil_img = PILImage.fromarray(image)
                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG", quality=85)
                jpeg_bytes = buf.getvalue()

            try:
                t0 = time.perf_counter()
                caption = self.ollama.caption_image(
                    jpeg_bytes, model=self.vlm_model, prompt=self.prompt
                )
                elapsed = time.perf_counter() - t0

                image_filename = ""
                if self.save_images:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    image_filename = f"{ts}.jpg"
                    image_path = os.path.join(self.image_save_dir, image_filename)
                    with open(image_path, "wb") as f:
                        f.write(jpeg_bytes)

                msg = String()
                if image_filename:
                    msg.data = f"[IMG:{image_filename}]{caption}"
                else:
                    msg.data = caption
                self.caption_publisher.publish(msg)
                self.logger.info(
                    f"Published caption ({elapsed:.1f}s, img={image_filename}): {caption[:100]}..."
                )
            except Exception as e:
                self.logger.error(f"Caption failed: {e}")

            last_caption_time = time.perf_counter()


def main(args=None):
    rclpy.init(args=args)
    node = CaptionerNode()
    node.start_caption_loop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.stop_caption_loop()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
