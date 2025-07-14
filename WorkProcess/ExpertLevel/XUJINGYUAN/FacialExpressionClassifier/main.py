import argparse
import torch
import torch.backends.cudnn as cudnn
from model import create_model
from dataset import create_data_loaders
from train_test import train_model, test_model
import os

def main():
    parser = argparse.ArgumentParser(description='MobileNetV3 面部表情识别训练')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['fer2013_original', 'fer2013_cleaned_original', 
                                'raf-db_original', 'raf-db_keypoint',
                                'affectnet_original', 'ferplus_original'],
                        help='选择数据集')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='批大小 (default: 32)')
    parser.add_argument('--scheduler', type=str, default='cosine_annealing',
                        help='scheduler (default: cosine_annealing)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数 (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率 (default: 0.001)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='数据加载器工作线程数 (default: 4)')
    parser.add_argument('--resume', type=str, default=None,
                        help='从检查点恢复训练')
    parser.add_argument('--test-only', action='store_true',
                        help='仅进行测试')
    
    args = parser.parse_args()
    
    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        cudnn.benchmark = True
    
    # 创建数据加载器
    print(f'加载数据集: {args.dataset}')
    train_loader, test_loader = create_data_loaders(
        args.dataset, 
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f'训练集大小: {len(train_loader.dataset)}')
    print(f'测试集大小: {len(test_loader.dataset)}')
    
    # 创建模型
    model = create_model(num_classes=7, pretrained=True)
    model = model.to(device)
    
    # 如果只进行测试
    if args.test_only:
        checkpoint_path = f'checkpoints/mobilenet_v3_{args.dataset}_epochs_{args.epochs}_scheduler_{args.scheduler}_lr_{args.lr}.pth'
        if os.path.exists(checkpoint_path):
            print(f'加载检查点: {checkpoint_path}')
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f'最佳验证准确率: {checkpoint["best_val_acc"]:.2f}%')
        else:
            print(f'检查点文件不存在: {checkpoint_path}')
            return
        
        # 测试模型并保存输出
        output_filename = f'mobilenet_v3_dataset_{args.dataset}_epochs_{args.epochs}_scheduler_{args.scheduler}_lr_{args.lr}.txt'
        test_acc, _, _ = test_model(model, test_loader, device, args.dataset, output_filename)
        return
    
    # 如果从检查点恢复训练
    start_epoch = 0
    best_val_acc = 0.0
    
    if args.resume:
        if os.path.exists(args.resume):
            print(f'从检查点恢复训练: {args.resume}')
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_acc = checkpoint['best_val_acc']
            print(f'从第 {start_epoch} 轮开始训练，最佳验证准确率: {best_val_acc:.2f}%')
        else:
            print(f'检查点文件不存在: {args.resume}')
    
    # 训练模型
    model, best_val_acc = train_model(
        model=model, 
        train_loader=train_loader, 
        val_loader=test_loader, 
        device=device, 
        args=args
    )
    
    # 加载最佳模型进行测试
    checkpoint_path = f'checkpoints/mobilenet_v3_{args.dataset}_epochs_{args.epochs}_scheduler_{args.scheduler}_lr_{args.lr}.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'加载最佳模型进行最终测试，验证准确率: {checkpoint["best_val_acc"]:.2f}%')
    
    # 最终测试
    output_filename = f'mobilenet_v3_{args.dataset}_epochs_{args.epochs}_scheduler_{args.scheduler}_lr_{args.lr}.txt'
    test_acc, _, _ = test_model(model, test_loader, device, args.dataset, output_filename)
    
    print(f'\n=== 训练完成 ===')
    print(f'数据集: {args.dataset}')
    print(f'最佳验证准确率: {best_val_acc:.2f}%')
    print(f'最终测试准确率: {test_acc:.2f}%')
    print(f'模型已保存至: checkpoints/mobilenet_v3_{args.dataset}.pth')

if __name__ == '__main__':
    main()
