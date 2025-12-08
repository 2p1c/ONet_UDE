"""
训练工具模块
"""

import torch
import torch.nn as nn
from typing import Tuple, List
import time


def train_epoch(model, dataloader, criterion, optimizer, device) -> float:
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
    
    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, criterion, device) -> float:
    """评估模型"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.item() * x.size(0)
    
    return total_loss / len(dataloader.dataset)


def train_model(
    model,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    device,
    epochs: int = 100,
    print_interval: int = 10,
    save_path: str = 'checkpoints/best_model.pth',
    # 【新增】早停参数
    early_stopping: bool = True,
    patience: int = 15,
    # 【新增】学习率调度参数
    use_scheduler: bool = True,
    scheduler_patience: int = 10,
    scheduler_factor: float = 0.5,
) -> Tuple[List[float], List[float], float]:
    """
    完整训练流程 - 增强版
    
    新增功能:
    - 早停法 (Early Stopping)
    - 学习率自适应调度 (ReduceLROnPlateau)
    
    Returns:
        train_losses, test_losses, best_test_loss
    """
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')
    
    # 【新增】早停计数器
    patience_counter = 0
    
    # 【修复】学习率调度器 - 移除 verbose 参数
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_factor,
            patience=scheduler_patience,
            min_lr=1e-6
        )
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # 测试
        test_loss = evaluate(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        
        # 【新增】学习率调度
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            old_lr = current_lr
            scheduler.step(test_loss)
            current_lr = optimizer.param_groups[0]['lr']
            # 【手动打印学习率变化】
            if old_lr != current_lr:
                print(f"   Learning rate reduced: {old_lr:.2e} → {current_lr:.2e}")
        
        # 保存最佳模型
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0  # 重置早停计数
            torch.save(model.state_dict(), save_path)
            improvement_flag = "✓"
        else:
            patience_counter += 1
            improvement_flag = ""
        
        # 打印进度
        if epoch % print_interval == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train: {train_loss:.6f} | "
                  f"Test: {test_loss:.6f} {improvement_flag} | "
                  f"LR: {current_lr:.2e} | "
                  f"Time: {elapsed:.1f}s")
        
        # 【新增】早停检查
        if early_stopping and patience_counter >= patience:
            print(f"\n⚠️ Early stopping triggered at epoch {epoch}")
            print(f"   No improvement for {patience} consecutive epochs")
            break
    
    total_time = time.time() - start_time
    print(f"\n✓ Training completed in {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"✓ Best test loss: {best_test_loss:.6f}")
    
    if early_stopping and patience_counter < patience:
        print(f"✓ Stopped at epoch {epoch}/{epochs} (no early stop)")
    
    return train_losses, test_losses, best_test_loss
