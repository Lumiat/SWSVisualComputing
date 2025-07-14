import argparse
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
import mediapipe as mp
from PIL import Image
import torchvision.transforms as transforms
from model import MobileNetV3ForExpression

class ExpressionTester:
    def __init__(self, model_path, device='cpu', input_size=48):
        """初始化表情识别测试器"""
        self.device = device
        self.model = self._load_model(model_path)
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        self.input_size = input_size
        
        # 表情标签和对应颜色
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.emotion_colors = {
            'surprise': 'orange',
            'angry': 'red', 
            'neutral': 'green',
            'happy': 'pink',
            'fear': 'mediumpurple',
            'disgust': 'olive',
            'sad': 'blue'
        }
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_model(self, model_path):
        """加载训练好的模型"""
        model = MobileNetV3ForExpression(num_classes=7, pretrained=False)
        
        # 加载模型权重
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        return model
    
    def detect_faces(self, image):
        """使用MediaPipe检测人脸"""
        # 转换为RGB格式
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)
        
        faces = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                
                # 转换为绝对坐标
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # 确保坐标在图像范围内
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)
                
                faces.append((x, y, width, height))
        
        return faces
    
    def predict_emotion(self, face_image):
        """预测人脸表情"""
        # 转换为PIL图像
        if len(face_image.shape) == 3:
            face_pil = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
        else:
            face_pil = Image.fromarray(face_image).convert('RGB')
        
        # 预处理
        face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(face_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_idx].item()
        
        return self.emotion_labels[predicted_idx], confidence
    
    def process_image(self, image_path):
        """处理单张图像"""
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return None
        
        # 检测人脸
        faces = self.detect_faces(image)
        
        results = []
        for face_bbox in faces:
            x, y, w, h = face_bbox
            
            # 提取人脸区域
            face_image = image[y:y+h, x:x+w]
            
            if face_image.size > 0:
                # 预测表情
                emotion, confidence = self.predict_emotion(face_image)
                results.append({
                    'bbox': face_bbox,
                    'emotion': emotion,
                    'confidence': confidence
                })
        
        return {
            'image': image,
            'faces': results,
            'image_path': image_path
        }
    
    def visualize_results(self, results, save_path='results'):
        """可视化结果"""
        if not results:
            print("没有结果可以可视化")
            return
        
        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)
        
        # 固定布局：4行5列
        rows = 4
        cols = 5
        max_images_per_page = rows * cols  # 每页最多显示20张图像
        
        # 计算需要多少页
        total_images = len(results)
        total_pages = (total_images + max_images_per_page - 1) // max_images_per_page
        
        print(f"总共 {total_images} 张图像，将分 {total_pages} 页显示")
        
        # 分页显示
        for page in range(total_pages):
            start_idx = page * max_images_per_page
            end_idx = min(start_idx + max_images_per_page, total_images)
            page_results = results[start_idx:end_idx]
            
            # 创建固定大小的子图
            fig, axes = plt.subplots(rows, cols, figsize=(20, 16))  # 调整图像大小
            
            # 设置整体标题
            if total_pages > 1:
                fig.suptitle(f'Expression Recognition Results - Page {page + 1}/{total_pages}', 
                           fontsize=16, fontweight='bold')
            else:
                fig.suptitle('Expression Recognition Results', fontsize=16, fontweight='bold')
            
            # 处理每个子图
            for idx in range(rows * cols):
                row = idx // cols
                col = idx % cols
                ax = axes[row, col]
                
                if idx < len(page_results):
                    # 有图像数据的子图
                    result = page_results[idx]
                    image = result['image']
                    faces = result['faces']
                    
                    # 转换为RGB用于matplotlib显示
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    ax.imshow(image_rgb)
                    
                    # 绘制人脸框和标签
                    title_emotions = []
                    for face in faces:
                        bbox = face['bbox']
                        emotion = face['emotion']
                        confidence = face['confidence']
                        
                        x, y, w, h = bbox
                        color = self.emotion_colors.get(emotion, 'yellow')
                        
                        # 绘制边框
                        rect = patches.Rectangle(
                            (x, y), w, h, 
                            linewidth=3,  # 增加边框宽度
                            edgecolor=color, 
                            facecolor='none'
                        )
                        ax.add_patch(rect)
                        
                        # 添加表情标签
                        ax.text(x, y-10, f'{emotion} ({confidence:.2f})', 
                               color=color, fontsize=10, fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                        
                        title_emotions.append(f"{emotion}({confidence:.2f})")
                    
                    # 设置标题
                    image_name = os.path.basename(result['image_path'])
                    if title_emotions:
                        title = f"{image_name}\n{', '.join(title_emotions)}"
                    else:
                        title = f"{image_name}\nNo face detected"
                    
                    ax.set_title(title, fontsize=12, pad=10)
                    ax.axis('off')
                else:
                    # 空的子图
                    ax.axis('off')
            
            # 调整子图间距
            plt.subplots_adjust(top=0.93, bottom=0.02, left=0.02, right=0.98, 
                              hspace=0.3, wspace=0.1)
            
            # 保存图像
            if total_pages > 1:
                save_file = os.path.join(save_path, f'{args.model}_{args.image}_page{page + 1}.png')
            else:
                save_file = os.path.join(save_path, f'{args.model}_{args.image}.png')
            
            plt.savefig(save_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.show()
            
            print(f"页面 {page + 1} 结果已保存到: {save_file}")
        
        print(f"所有 {total_images} 张图像处理完成，分 {total_pages} 页显示")
    
    def run_test(self, image):
        """运行测试"""
        # 确定图像文件夹路径
        if image == 'grayscale':
            image_folder = './test_images/grayscale'
        elif image == '3channels':
            image_folder = './test_images/3channels'
        elif image == 'raf-db_val':
            image_folder = './test_images/raf-db_val'
        else:
            raise ValueError("image参数必须是'grayscale'或'3channels'")
        
        if not os.path.exists(image_folder):
            print(f"图像文件夹不存在: {image_folder}")
            return
        
        # 获取所有图像文件并排序
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for file in sorted(os.listdir(image_folder)):  # 添加排序确保顺序一致
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(image_folder, file))
        
        if not image_files:
            print(f"在文件夹 {image_folder} 中没有找到图像文件")
            return
        
        print(f"找到 {len(image_files)} 张图像")
        
        # 处理所有图像
        results = []
        for image_path in image_files:
            print(f"处理图像: {os.path.basename(image_path)}")
            result = self.process_image(image_path)
            if result:
                results.append(result)
        
        # 可视化结果
        if results:
            self.visualize_results(results)
            print(f"完成处理 {len(results)} 张图像")
        else:
            print("没有成功处理任何图像")

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='表情识别测试脚本')
    parser.add_argument('--image', choices=['grayscale', '3channels', 'raf-db_val'], 
                       required=True, help='选择图像类型: grayscale, 3channels or raf-db_val')
    parser.add_argument('--model', choices=['fer2013', 'raf-db', 'fer2013_cleaned', 'affectnet'], 
                       required=True, help='choose model: fer2013, raf-db, fer2013_cleaned or affectnet')
    
    global args
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if args.model == 'fer2013':
        model_path = './checkpoints/mobilenet_v3_fer2013_original.pth'
        input_size = 48
    elif args.model == 'raf-db':
        model_path = './checkpoints/mobilenet_v3_raf-db_original.pth'
        input_size = 75
    elif args.model == 'fer2013_cleaned':
        model_path = './checkpoints/mobilenet_v3_fer2013_cleaned_original_epochs_100_scheduler_cosine_annealing_lr_0.001.pth'
        input_size = 48
    elif args.model == 'affectnet':
        model_path = './checkpoints/mobilenet_v3_affectnet_original_epochs_100_scheduler_cosine_annealing_lr_0.001.pth'
        input_size = 96
    
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建测试器并运行
    tester = ExpressionTester(model_path, device, input_size)
    tester.run_test(args.image)

if __name__ == "__main__":
    main()
