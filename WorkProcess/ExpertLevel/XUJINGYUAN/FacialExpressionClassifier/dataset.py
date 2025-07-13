import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

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

def get_dataset_class(dataset_name):
    """根据数据集名称获取对应的数据集类"""
    dataset_classes = {
        'fer2013_original': FER2013OriginalDataset,
        'fer2013_cleaned_original': FER2013CleanedOriginalDataset,
        'raf-db_original': RAFDBOriginalDataset,
        'raf-db_keypoint': RAFDBKeypointDataset,
        'affectnet_original': AffectNetDataset
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
