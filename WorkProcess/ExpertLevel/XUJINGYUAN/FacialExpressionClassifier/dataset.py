import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd


class BaseDataset(Dataset):
    """基础数据集加载类"""
    
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # 情绪类别映射
        self.emotion_labels = {
            'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3,
            'sad': 4, 'surprise': 5, 'neutral': 6
        }
        
        # 加载数据路径和标签
        self.data_paths = []
        self.labels = []
        self._load_data()
    
    def _load_data(self):
        """加载数据路径和标签"""
        split_dir = os.path.join(self.root_dir, self.split)
        
        for emotion_name, label in self.emotion_labels.items():
            emotion_dir = os.path.join(split_dir, emotion_name)
            if os.path.exists(emotion_dir):
                for img_name in os.listdir(emotion_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(emotion_dir, img_name)
                        self.data_paths.append(img_path)
                        self.labels.append(label)
    
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        img_path = self.data_paths[idx]
        label = self.labels[idx]
        
        # 加载图像并转换为RGB三通道
        image = Image.open(img_path)
        if image.mode != 'RGB':
            # 如果是灰度图，先转换为L模式，然后转换为RGB
            if image.mode != 'L':
                image = image.convert('L')
            image = image.convert('RGB')  # 将灰度图转换为RGB三通道
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    @staticmethod
    def get_transforms(split='train', input_size=224):
        """获取数据预处理transforms"""
        if split == 'train':
            return transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                # 使用RGB图像的标准归一化参数
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

class FER2013OriginalDataset(BaseDataset):
    """FER2013 Original数据集"""
    
    def __init__(self, root_dir, split='train', transform=None):
        super().__init__(root_dir, split, transform)
        
    @staticmethod
    def get_transforms(split='train'):
        return BaseDataset.get_transforms(split, input_size=48)

class FER2013CleanedOriginalDataset(BaseDataset):
    """FER2013 CleanedOriginal数据集"""
    
    def __init__(self, root_dir, split='train', transform=None):
        super().__init__(root_dir, split, transform)
        
    @staticmethod
    def get_transforms(split='train'):
        return BaseDataset.get_transforms(split, input_size=48)

class RAFDBOriginalDataset(BaseDataset):
    """RAF-DB Original数据集"""
    
    def __init__(self, root_dir, split='train', transform=None):
        super().__init__(root_dir, split, transform)
        
    @staticmethod
    def get_transforms(split='train'):
        return BaseDataset.get_transforms(split, input_size=75)

class RAFDBKeypointDataset(BaseDataset):
    """RAF-DB Keypoint数据集"""
    
    def __init__(self, root_dir, split='train', transform=None):
        super().__init__(root_dir, split, transform)
        
    @staticmethod
    def get_transforms(split='train'):
        return BaseDataset.get_transforms(split, input_size=75)
    
class AffectNetDataset(BaseDataset):
    """RAF-DB Keypoint数据集"""
    
    def __init__(self, root_dir, split='train', transform=None):
        super().__init__(root_dir, split, transform)
        
    @staticmethod
    def get_transforms(split='train'):
        return BaseDataset.get_transforms(split, input_size=96)

class FERPlusDataset(BaseDataset):
    """FERPlus数据集，支持多标注者软标签"""
    
    def __init__(self, root_dir, split='train', transform=None, 
                 use_soft_labels=True, temperature=2.0, consistency_weight=True):
        self.use_soft_labels = use_soft_labels
        self.temperature = temperature
        self.consistency_weight = consistency_weight
        
        # FERPlus的情绪映射 (不包含contempt)
        self.emotion_labels = {
            'neutral': 0, 'happy': 1, 'surprise': 2, 'sad': 3,
            'angry': 4, 'disgust': 5, 'fear': 6
        }
        
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # 加载数据路径和标签
        self.data_paths = []
        self.labels = []
        self.soft_labels = []
        self.annotation_counts = []
        
        self._load_ferplus_data()
    
    def _load_ferplus_data(self):
        """加载FERPlus数据"""
        # 根据split确定文件夹名称
        split_mapping = {
            'train': 'train',
            'test': 'test', 
            'val': 'validate',
            'validate': 'validate'
        }
        
        split_folder = split_mapping.get(self.split, self.split)
        
        # 图像文件夹路径
        images_dir = os.path.join(self.root_dir, split_folder, 'images')
        
        # 标签文件路径
        label_file = os.path.join(self.root_dir, split_folder, f'{split_folder}_label.csv')
        
        if not os.path.exists(label_file):
            raise FileNotFoundError(f"标签文件不存在: {label_file}")
        
        # 读取标签文件
        df = pd.read_csv(label_file)
        
        # FERPlus的列名映射
        ferplus_columns = ['neutral', 'happy', 'surprise', 'sad', 'angry', 'disgust', 'fear']
        
        for _, row in df.iterrows():
            image_name = row['Image name']
            image_path = os.path.join(images_dir, image_name)
            
            # 检查图像文件是否存在
            if not os.path.exists(image_path):
                continue
            
            # 提取标注者投票
            annotation_votes = []
            for col in ferplus_columns:
                if col in row:
                    annotation_votes.append(row[col])
                else:
                    annotation_votes.append(0)
            
            annotation_votes = np.array(annotation_votes)
            
            # 过滤掉所有投票都为0的样本
            if annotation_votes.sum() == 0:
                continue
            
            # 生成硬标签（多数投票）
            hard_label = np.argmax(annotation_votes)
            
            # 生成软标签
            if self.use_soft_labels:
                soft_label = self._generate_soft_label(annotation_votes)
            else:
                soft_label = np.zeros(7)
                soft_label[hard_label] = 1.0
            
            self.data_paths.append(image_path)
            self.labels.append(hard_label)
            self.soft_labels.append(soft_label)
            self.annotation_counts.append(annotation_votes)
    
    def _generate_soft_label(self, annotation_votes):
        """生成软标签"""
        # 计算总投票数
        total_votes = annotation_votes.sum()
        if total_votes == 0:
            return np.ones(7) / 7  # 均匀分布
        
        # 基础概率分布
        base_probs = annotation_votes / total_votes
        
        # 如果使用一致性权重
        if self.consistency_weight:
            # 计算一致性权重
            consistency = self._calculate_consistency(annotation_votes)
            # 调整软标签的确定性
            adjusted_probs = base_probs * (1 + consistency)
            adjusted_probs = adjusted_probs / adjusted_probs.sum()
        else:
            adjusted_probs = base_probs
        
        # 应用温度参数
        if self.temperature != 1.0:
            # 避免log(0)
            adjusted_probs = np.maximum(adjusted_probs, 1e-8)
            log_probs = np.log(adjusted_probs) / self.temperature
            soft_probs = np.exp(log_probs)
            soft_probs = soft_probs / soft_probs.sum()
        else:
            soft_probs = adjusted_probs
        
        return soft_probs
    
    def _calculate_consistency(self, annotation_votes):
        """计算标注一致性"""
        total_votes = annotation_votes.sum()
        if total_votes <= 1:
            return 0.0
        
        # 使用熵来衡量一致性
        probs = annotation_votes / total_votes
        probs = probs[probs > 0]  # 避免log(0)
        
        if len(probs) <= 1:
            return 1.0  # 完全一致
        
        entropy = -np.sum(probs * np.log(probs))
        max_entropy = np.log(len(probs))
        
        # 一致性 = 1 - 归一化熵
        consistency = 1 - (entropy / max_entropy) if max_entropy > 0 else 1.0
        
        return consistency
    
    def __getitem__(self, idx):
        img_path = self.data_paths[idx]
        hard_label = self.labels[idx]
        soft_label = self.soft_labels[idx]
        
        # 加载图像
        image = Image.open(img_path)
        if image.mode != 'RGB':
            if image.mode != 'L':
                image = image.convert('L')
            image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, hard_label, torch.FloatTensor(soft_label)
    
    @staticmethod
    def get_transforms(split='train'):
        return BaseDataset.get_transforms(split, input_size=48)
    
    def get_annotation_statistics(self):
        """获取标注统计信息"""
        if not self.annotation_counts:
            return None
        
        annotation_counts = np.array(self.annotation_counts)
        
        stats = {
            'total_samples': len(self.annotation_counts),
            'total_annotations': annotation_counts.sum(),
            'avg_annotations_per_sample': annotation_counts.sum(axis=1).mean(),
            'emotion_distribution': annotation_counts.sum(axis=0),
            'emotion_names': list(self.emotion_labels.keys())
        }
        
        return stats

def get_dataset_class(dataset_name):
    """根据数据集名称获取对应的数据集类"""
    dataset_classes = {
        'fer2013_original': FER2013OriginalDataset,
        'fer2013_cleaned_original': FER2013CleanedOriginalDataset,
        'raf-db_original': RAFDBOriginalDataset,
        'raf-db_keypoint': RAFDBKeypointDataset,
        'affectnet_original': AffectNetDataset,
        'ferplus_original': FERPlusDataset
    }
    return dataset_classes.get(dataset_name)

def create_data_loaders(dataset_name, batch_size=32, num_workers=4):
    """创建数据加载器"""
    dataset_class = get_dataset_class(dataset_name)
    if dataset_class is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # 数据集路径
    dataset_path = os.path.join('dataset', dataset_name)
    
    # 创建数据集
    train_transform = dataset_class.get_transforms('train')
    test_transform = dataset_class.get_transforms('test')
    
    if dataset_name == 'ferplus_original':
        train_dataset = dataset_class(
            root_dir=dataset_path,
            split='train',
            transform=train_transform,
            use_soft_labels=True,
            temperature=2.0,
            consistency_weight=True
        )
        
        test_dataset = dataset_class(
            root_dir=dataset_path,
            split='test',
            transform=test_transform,
            use_soft_labels=True,
            temperature=2.0,
            consistency_weight=True
        )
    else:
        train_dataset = dataset_class(
            root_dir=dataset_path,
            split='train',
            transform=train_transform
        )
        
        test_dataset = dataset_class(
            root_dir=dataset_path,
            split='test',
            transform=test_transform
        )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader
