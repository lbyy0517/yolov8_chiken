import cv2
import numpy as np
import os
from pathlib import Path
import argparse

class VideoPreprocessor:
    
    def __init__(self, video_path, output_dir, frame_interval=30):
        #video_path: MP4视频文件路径；output_dir: 输出文件夹路径；frame_interval: 抽帧间隔（每隔多少帧提取一帧）
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.frame_interval = frame_interval
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_frames(self):       
        # 打开视频文件
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {self.video_path}")
        
        # 获取视频信息，total_frames:视频总帧数；fps:视频帧率；frame_interval:抽帧间隔
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  #
        fps = cap.get(cv2.CAP_PROP_FPS)

        frame_count = 0
        saved_count = 0
        saved_files = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 按间隔提取帧
            if frame_count % self.frame_interval == 0:
                # 生成文件名
                timestamp = frame_count / fps
                filename = f"frame_{saved_count:06d}_t{timestamp:.2f}s.jpg"
                filepath = self.output_dir / filename
                
                # 保存原始帧
                cv2.imwrite(str(filepath), frame)
                saved_files.append(filepath)
                saved_count += 1
            frame_count += 1
        
        cap.release()
        print(f"共提取 {saved_count} 帧图片")
        return saved_files
    
    def add_gaussian_noise(self, image, noise_level=25):
        #image: 输入图像；noise_level: 噪声强度 (0-100)
        # 生成高斯噪声
        noise = np.random.normal(0, noise_level, image.shape).astype(np.float32)
        
        # 添加噪声
        noisy_image = image.astype(np.float32) + noise
        
        # 限制像素值范围到 [0, 255]
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        
        return noisy_image
    
    def adjust_exposure(self, image, exposure_factor=1.2):
        #image: 输入图像；exposure_factor: 曝光调整因子 (>1 增加曝光, <1 减少曝光)
        # 转换为浮点数进行计算
        adjusted = image.astype(np.float32) * exposure_factor
        
        # 限制像素值范围
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        
        return adjusted
    
    def process_images(self, image_files, noise_levels=[15, 25, 35], exposure_factors=[0.8, 1.2, 1.5]):
        #image_files: 图像文件路径列表；noise_levels: 噪声水平列表；exposure_factors: 曝光调整因子列表
        for i, image_file in enumerate(image_files):
            # 读取原始图像
            original_image = cv2.imread(str(image_file))
            if original_image is None:
                print(f"警告: 无法读取图像 {image_file}")
                continue
            
            # 获取原始文件名（不含扩展名）
            base_name = image_file.stem
            
            # 生成噪声版本
            for noise_level in noise_levels:
                noisy_image = self.add_gaussian_noise(original_image, noise_level)
                noise_filename = f"{base_name}_noise{noise_level}.jpg"
                noise_filepath = self.output_dir / noise_filename
                cv2.imwrite(str(noise_filepath), noisy_image)
            
            # 生成曝光调整版本
            for exposure_factor in exposure_factors:
                exposed_image = self.adjust_exposure(original_image, exposure_factor)
                exposure_filename = f"{base_name}_exp{exposure_factor:.1f}.jpg"
                exposure_filepath = self.output_dir / exposure_filename
                cv2.imwrite(str(exposure_filepath), exposed_image)
            
            # 生成组合版本（噪声+曝光）
            for noise_level in noise_levels:
                for exposure_factor in exposure_factors:
                    # 先添加噪声，再调整曝光
                    combined_image = self.add_gaussian_noise(original_image, noise_level)
                    combined_image = self.adjust_exposure(combined_image, exposure_factor)
                    
                    combined_filename = f"{base_name}_noise{noise_level}_exp{exposure_factor:.1f}.jpg"
                    combined_filepath = self.output_dir / combined_filename
                    cv2.imwrite(str(combined_filepath), combined_image)

    
    def run_full_pipeline(self, noise_levels=[15, 25, 35], exposure_factors=[0.8, 1.2, 1.5]):
        #noise_levels: 噪声水平列表；exposure_factors: 曝光调整因子列表
        # 步骤1: 提取帧
        extracted_files = self.extract_frames()
        
        # 步骤2: 图像预处理
        if extracted_files:
            self.process_images(extracted_files, noise_levels, exposure_factors)

if __name__ == "__main__":
    # 方法1: 直接在代码中使用
    preprocessor = VideoPreprocessor(
        video_path="C:/Users/zemyee/Desktop/test.mp4",
        output_dir="C:/Users/zemyee/Desktop/test",
        frame_interval=30
        )
    preprocessor.run_full_pipeline()
    

"""
输出文件命名规则:
- 原始帧: frame_000001_t1.23s.jpg
- 噪声版本: frame_000001_t1.23s_noise25.jpg  
- 曝光版本: frame_000001_t1.23s_exp1.2.jpg
- 组合版本: frame_000001_t1.23s_noise25_exp1.2.jpg
"""