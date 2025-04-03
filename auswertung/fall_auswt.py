import torch
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc
)
import matplotlib.pyplot as plt
import seaborn as sns



def evaluate_model(model, test_loader, device):
    model.eval()  # 设置模型为评估模式
    num_correct = 0.0
    total_samples = 0
    
    # 用于存储混淆矩阵的数据
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for test_data in test_loader:
            test_images, test_labels = test_data[0].to(device), test_data[1].to(device)
            
            # 模型预测
            test_outputs = model(test_images)
            
            # 计算准确率
            preds = test_outputs.argmax(dim=1)
            true_labels = test_labels.argmax(dim=1)
            
            num_correct += (preds == true_labels).sum().item()
            total_samples += test_labels.size(0)
            
            # 收集预测和真实标签用于后续分析
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(true_labels.cpu().numpy())
    
    # 计算准确率
    test_accuracy = num_correct / total_samples
    
    # 使用 sklearn 计算更详细的指标
    from sklearn.metrics import classification_report, confusion_matrix
    print("Classification Report:")
    print(classification_report(all_labels, all_preds))
    
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    
    return test_accuracy



def plot_confusion_matrix(model, test_loader, device, classes):
    model.eval()  # 设置模型为评估模式
    num_correct = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for test_data in test_loader:
            test_images, test_labels = test_data[0].to(device), test_data[1].to(device)
            
            # 模型预测
            test_outputs = model(test_images)
            
            # 计算准确率
            preds = test_outputs.argmax(dim=1)
            true_labels = test_labels.argmax(dim=1)
            
            num_correct += (preds == true_labels).sum().item()
            total_samples += test_labels.size(0)
            
            # 收集预测和真实标签用于后续分析
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(true_labels.cpu().numpy())
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, 
                yticklabels=classes)
    plt.title('Title')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()



def plot_roc_bin(model, test_loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for test_data in test_loader:
            test_images, test_labels = test_data[0].to(device), test_data[1].to(device)
            test_outputs = torch.softmax(model(test_images), dim=1)
            
            all_probs.extend(test_outputs[:, 1].cpu().numpy())
            all_labels.extend(test_labels[:, 1].cpu().numpy())
    
    # 计算 ROC 曲线
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

