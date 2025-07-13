import cv2
import mediapipe as mp
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
from model import MobileNetV3ForExpression  # 导入你的模型类
import os
import argparse
from collections import deque, Counter

# MediaPipe 初始化 - 使用人脸检测而不是人脸网格
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# 表情标签和对应颜色
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral']
EMOTION_COLORS = {
    'angry': (0, 0, 255),      # 红色
    'sad': (255, 0, 0),        # 蓝色
    'neutral': (0, 255, 0),    # 绿色
    'happy': (255, 0, 255),    # 粉色
    'disgust': (128, 0, 128),  # 紫色
    'surprised': (0, 165, 255), # 橙色
    'fear': (255, 255, 0)      # 黄色
}

class EmotionStabilizer:
    """
    表情识别结果稳定器，用于减少闪动
    """
    def __init__(self, window_size=3, mode='confidence'):
        """
        初始化稳定器
        
        Args:
            window_size (int): 滑动窗口大小，默认为5帧
            mode (str): 判断模式，'confidence' 或 'frequency'
        """
        self.window_size = window_size
        self.mode = mode
        self.predictions = deque(maxlen=window_size)  # 存储最近的预测结果
        
    def add_prediction(self, emotion, confidence):
        """
        添加新的预测结果
        
        Args:
            emotion (str): 预测的表情
            confidence (float): 置信度
        """
        self.predictions.append((emotion, confidence))
    
    def get_stable_prediction(self):
        """
        获取稳定的预测结果
        
        Returns:
            tuple: (稳定的表情标签, 对应的置信度)
        """
        if not self.predictions:
            return 'neutral', 0.0
        
        if self.mode == 'confidence':
            # 置信度最高模式：找到置信度最高的预测结果
            best_prediction = max(self.predictions, key=lambda x: x[1])
            return best_prediction
        
        elif self.mode == 'frequency':
            # 出现次数最多模式：找到出现次数最多的表情
            emotions = [pred[0] for pred in self.predictions]
            emotion_counts = Counter(emotions)
            
            # 找到出现次数最多的表情
            most_common_emotion = emotion_counts.most_common(1)[0][0]
            
            # 找到该表情对应的最高置信度
            confidence_for_emotion = [pred[1] for pred in self.predictions if pred[0] == most_common_emotion]
            max_confidence = max(confidence_for_emotion)
            
            return most_common_emotion, max_confidence
        
        else:
            # 默认返回置信度最高的
            best_prediction = max(self.predictions, key=lambda x: x[1])
            return best_prediction
    
    def is_ready(self):
        """
        检查是否有足够的帧数进行稳定预测
        
        Returns:
            bool: 是否准备好
        """
        return len(self.predictions) >= min(3, self.window_size)  # 至少需要3帧或窗口大小的帧数

class EmotionRecognizer:
    def __init__(self, model_path, device='cpu', input_size=48):
        """
        初始化表情识别器
        
        Args:
            model_path (str): 训练好的模型文件路径
            device (str): 运行设备 ('cpu' 或 'cuda')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 创建模型实例
        self.model = MobileNetV3ForExpression(num_classes=7, pretrained=False)
        
        # 加载模型权重
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 从checkpoint中提取模型状态字典
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"模型加载成功，验证准确率: {checkpoint.get('best_val_acc', 'N/A'):.2f}%")
        else:
            # 如果直接保存的是模型状态字典
            self.model.load_state_dict(checkpoint)
            print("模型加载成功")
        
        # 将模型移动到指定设备并设置为评估模式
        self.model.to(self.device)
        self.model.eval()
        
        # 图像预处理 - 根据你的模型训练方式调整
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),  # 根据你的模型输入尺寸
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet标准化
        ])
    
    def predict_emotion(self, face_image):
        """
        预测人脸表情
        
        Args:
            face_image (numpy.ndarray): 人脸图像
            
        Returns:
            tuple: (预测的表情标签, 置信度)
        """
        try:
            # 转换为PIL图像
            pil_image = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
            
            # 预处理
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # 推理
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                emotion_label = EMOTION_LABELS[predicted.item()]
                confidence_score = confidence.item()
                
                return emotion_label, confidence_score
                
        except Exception as e:
            print(f"表情识别出错: {e}")
            return 'neutral', 0.0

def get_face_bbox_from_detection(detection, image_shape):
    """
    从MediaPipe检测结果获取人脸边界框
    
    Args:
        detection: MediaPipe人脸检测结果
        image_shape: 图像尺寸 (height, width, channels)
        
    Returns:
        tuple: (x1, y1, x2, y2) 边界框坐标
    """
    h, w = image_shape[:2]
    
    # 获取相对坐标的边界框
    bbox = detection.location_data.relative_bounding_box
    
    # 转换为绝对坐标
    x1 = int(bbox.xmin * w)
    y1 = int(bbox.ymin * h)
    x2 = int((bbox.xmin + bbox.width) * w)
    y2 = int((bbox.ymin + bbox.height) * h)
    
    # 确保坐标在图像范围内
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    return x1, y1, x2, y2

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='apply real-time facial expression recognition')
    parser.add_argument('--model', choices=['fer2013', 'raf-db', 'fer2013_cleaned', 'affectnet'], 
                       required=True, help='choose model: fer2013, raf-db, fer2013_cleaned or affectnet')
    parser.add_argument('--window_size', type=int, default=5, 
                       help='滑动窗口大小，用于稳定表情识别结果 (default: 5)')
    parser.add_argument('--mode', choices=['confidence', 'frequency'], default='confidence',
                       help='表情判断模式: confidence (置信度最高) 或 frequency (出现次数最多) (default: confidence)')
    
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if args.model == 'fer2013':
        MODEL_PATH = './checkpoints/mobilenet_v3_fer2013_original.pth'
        INPUT_SIZE = 48
    elif args.model == 'raf-db':
        MODEL_PATH = './checkpoints/mobilenet_v3_raf-db_original.pth'
        INPUT_SIZE = 75
    elif args.model == 'fer2013_cleaned':
        MODEL_PATH = './checkpoints/mobilenet_v3_fer2013_cleaned_original_epochs_100_scheduler_cosine_annealing_lr_0.001.pth'
        INPUT_SIZE = 48
    elif args.model == 'affectnet':
        MODEL_PATH = './checkpoints/mobilenet_v3_affectnet_original_epochs_100_scheduler_cosine_annealing_lr_0.0005.pth'
        INPUT_SIZE = 96
        
    if not os.path.exists(MODEL_PATH):
        print(f"模型文件不存在: {MODEL_PATH}")
        return
    
    try:
        emotion_recognizer = EmotionRecognizer(MODEL_PATH, input_size=INPUT_SIZE)
        print("表情识别器初始化成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 初始化表情稳定器
    emotion_stabilizer = EmotionStabilizer(window_size=args.window_size, mode=args.mode)
    mode_desc = "置信度最高" if args.mode == 'confidence' else "出现次数最多"
    print(f"表情稳定器初始化成功，窗口大小: {args.window_size}, 模式: {mode_desc}")
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    print("摄像头已打开，按 'q' 键退出")
    
    # 使用人脸检测而不是人脸网格
    with mp_face_detection.FaceDetection(
            model_selection=0,  # 0 用于近距离人脸检测（2米内）
            min_detection_confidence=0.5) as face_detection:
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("无法读取摄像头画面")
                break
            
            # 水平翻转图像 (镜像效果)
            frame = cv2.flip(frame, 1)
            
            # 转换颜色空间
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = face_detection.process(image)
            
            # 恢复颜色空间
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.detections:
                for detection in results.detections:
                    # 获取人脸边界框
                    x1, y1, x2, y2 = get_face_bbox_from_detection(detection, image.shape)
                    
                    # 提取人脸区域进行表情识别
                    face_roi = image[y1:y2, x1:x2]
                    
                    if face_roi.size > 0:
                        # 预测表情
                        emotion, confidence = emotion_recognizer.predict_emotion(face_roi)
                        
                        # 添加到稳定器
                        emotion_stabilizer.add_prediction(emotion, confidence)
                        
                        # 获取稳定的预测结果
                        if emotion_stabilizer.is_ready():
                            stable_emotion, stable_confidence = emotion_stabilizer.get_stable_prediction()
                        else:
                            # 如果还没有足够的帧数，使用当前预测结果
                            stable_emotion, stable_confidence = emotion, confidence
                        
                        # 获取对应颜色
                        color = EMOTION_COLORS.get(stable_emotion, (0, 255, 0))
                        
                        # 绘制边界框
                        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                        
                        # 显示表情标签和置信度
                        text = f"{stable_emotion}: {stable_confidence:.2f}"
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        
                        # 绘制文本背景
                        cv2.rectangle(image, (x1, y1-30), (x1+text_size[0]+10, y1), color, -1)
                        
                        # 绘制文本
                        cv2.putText(image, text, (x1+5, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # 显示当前帧的原始预测结果（用于调试）
                        debug_text = f"Raw: {emotion}: {confidence:.2f}"
                        cv2.putText(image, debug_text, (x1+5, y2+20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
                        
                        # 显示当前使用的模式
                        mode_text = f"Mode: {args.mode}"
                        cv2.putText(image, mode_text, (x1+5, y2+40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
            else:
                # 如果没有检测到人脸，清空稳定器
                emotion_stabilizer.predictions.clear()
            
            # 显示图像
            cv2.imshow('Real-time Emotion Recognition', image)
            
            # 按'q'键退出
            if (cv2.waitKey(1) & 0xFF == ord('q')) or (cv2.getWindowProperty('Real-time Emotion Recognition', cv2.WND_PROP_VISIBLE) < 1):
                break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("程序已退出")

if __name__ == "__main__":
    main()
