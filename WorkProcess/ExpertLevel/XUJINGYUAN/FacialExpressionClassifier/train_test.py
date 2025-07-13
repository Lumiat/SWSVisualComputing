import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import swanlab
import os
import time
from tqdm import tqdm
import io

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} Training')
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate_epoch(model, val_loader, criterion, device, epoch):
    """验证一个epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} Validation')
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc, all_predictions, all_targets

def train_model(model, train_loader, val_loader, device, args):
    """训练模型"""
    
    # 初始化SwanLab
    swanlab.init(
        project="facial-expression-recognition",
        experiment_name=f"mobilenet_v3_{args.dataset}",
        config={
            "model": "MobileNetV3",
            "dataset": args.dataset,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "batch_size": train_loader.batch_size,
        }
    )
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if args.scheduler == 'reduce_lr_on_plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    elif args.scheduler == 'cosine_annealing':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    
    # 训练记录
    best_val_acc = 0.0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # 创建保存目录
    save_dir = 'checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"开始训练 MobileNetV3 模型，使用数据集: {args.dataset}")
    print(f"模型参数数量: {model.count_parameters():,}")
    print(f"模型大小: {model.get_model_size():.2f} MB")
    
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        print('-' * 50)
        
        # 训练阶段
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # 验证阶段
        val_loss, val_acc, val_predictions, val_targets = validate_epoch(
            model, val_loader, criterion, device, epoch
        )
        
        # 学习率调整
        if args.scheduler == 'reduce_lr_on_plateau':
            scheduler.step(val_loss)
        elif args.scheduler == 'cosine_annealing':
            scheduler.step()
        
        # 记录训练数据
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 记录到SwanLab
        swanlab.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'train_losses': train_losses,
                'train_accs': train_accs,
                'val_losses': val_losses,
                'val_accs': val_accs
            }
            
            checkpoint_path = os.path.join(save_dir, f'mobilenet_v3_{args.dataset}_epochs_{args.epochs}_scheduler_{args.scheduler}_lr_{args.lr}.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f'✓ 保存最佳模型，验证准确率: {best_val_acc:.2f}%')
        
        print(f'训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%')
        print(f'验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%')
        print(f'当前学习率: {optimizer.param_groups[0]["lr"]:.6f}')
    
    # 生成最终的分类报告
    emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    final_report = classification_report(val_targets, val_predictions, 
                                       target_names=emotion_names, 
                                       output_dict=True)
    
    # 记录最终结果到SwanLab
    swanlab.log({
        "final_val_accuracy": best_val_acc,
        "classification_report": final_report
    })
    
    print(f'\n训练完成！最佳验证准确率: {best_val_acc:.2f}%')
    print(f'模型已保存至: checkpoints/mobilenet_v3_{args.dataset}_epochs_{args.epochs}_scheduler_{args.scheduler}_lr_{args.lr}.pth')
    
    swanlab.finish()
    
    return model, best_val_acc

def test_model(model, test_loader, device, dataset_name, output_filename=None):
    """测试模型"""
    
    # 创建一个StringIO对象来捕获输出
    output_buffer = io.StringIO()
    
    # 获取当前时间戳
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 定义一个函数来同时打印到控制台和缓冲区
    def tee_print(*args, **kwargs):
        print(*args, **kwargs)
        print(*args, file=output_buffer, **kwargs)
    
    tee_print(f"\n{'='*60}")
    tee_print(f"模型测试报告")
    tee_print(f"生成时间: {timestamp}")
    tee_print(f"数据集: {dataset_name}")
    tee_print(f"{'='*60}")
    
    tee_print(f"\n开始测试模型，使用数据集: {dataset_name}")
    
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    # 测试开始时间
    start_time = time.time()
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            pbar.set_postfix({'Acc': f'{100.*correct/total:.2f}%'})
    
    # 测试结束时间
    end_time = time.time()
    test_time = end_time - start_time
    
    test_acc = 100. * correct / total
    
    # 生成详细的分类报告
    emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    tee_print(f'\n测试结果:')
    tee_print(f'测试样本数量: {total}')
    tee_print(f'正确预测数量: {correct}')
    tee_print(f'测试准确率: {test_acc:.2f}%')
    tee_print(f'测试耗时: {test_time:.2f} 秒')
    tee_print(f'平均每个样本处理时间: {test_time/total*1000:.2f} 毫秒')
    
    tee_print('\n详细分类报告:')
    classification_report_str = classification_report(all_targets, all_predictions, target_names=emotion_names)
    tee_print(classification_report_str)
    
    # 混淆矩阵
    cm = confusion_matrix(all_targets, all_predictions)
    tee_print('\n混淆矩阵:')
    tee_print('实际\\预测', end='')
    for emotion in emotion_names:
        tee_print(f'{emotion:>10}', end='')
    tee_print()
    
    for i, emotion in enumerate(emotion_names):
        tee_print(f'{emotion:>10}', end='')
        for j in range(len(emotion_names)):
            tee_print(f'{cm[i][j]:>10}', end='')
        tee_print()
    
    # 每个类别的准确率
    tee_print('\n各类别准确率:')
    for i, emotion in enumerate(emotion_names):
        class_correct = cm[i][i]
        class_total = cm[i].sum()
        class_acc = class_correct / class_total * 100 if class_total > 0 else 0
        tee_print(f'{emotion:>10}: {class_acc:>6.2f}% ({class_correct:>3}/{class_total:>3})')
    
    tee_print(f'\n{"="*60}')
    tee_print(f"测试完成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    tee_print(f'{"="*60}')
    
    # 如果指定了输出文件名，则保存到文件
    if output_filename:
        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                f.write(output_buffer.getvalue())
            print(f'\n测试结果已保存至: {output_filename}')
        except Exception as e:
            print(f'\n保存测试结果失败: {e}')
    
    return test_acc, all_predictions, all_targets