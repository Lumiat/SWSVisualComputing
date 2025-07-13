import torch
import torch.nn as nn
import timm

class MobileNetV3ForExpression(nn.Module):
    def __init__(self, num_classes=7, model_name='mobilenetv3_small_100', pretrained=True):
        super(MobileNetV3ForExpression, self).__init__()
        
        # 加载预训练的MobileNetV3模型
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # 移除原始分类头
            global_pool='avg'
        )
        
        # 获取特征维度
        self.feature_dim = self._get_feature_dim()
        print(f"d_features: {self.feature_dim}")
        
        # 自定义分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
        # 初始化分类头权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 提取特征
        features = self.backbone(x)
        
        # 分类
        output = self.classifier(features)
        
        return output
    
    def get_model_size(self):
        """获取模型大小（MB）"""
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def count_parameters(self):
        """计算模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _get_feature_dim(self):
            """动态获取特征维度"""
            # 使用48x48的输入尺寸（匹配您的数据）
            test_input = torch.randn(1, 3, 48, 48)
            self.backbone.eval()
            
            with torch.no_grad():
                features = self.backbone(test_input)
                feature_dim = features.shape[1]
            
            return feature_dim

def create_model(num_classes=7, pretrained=True):
    """创建模型实例"""
    model = MobileNetV3ForExpression(
        num_classes=num_classes, 
        pretrained=pretrained
    )
    
    return model
