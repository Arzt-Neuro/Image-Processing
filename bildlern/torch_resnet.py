import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


# 定义结合CT图像和临床特征的ResNet模型
class BrainHemorrhageModel(nn.Module):
    def __init__(self, num_clinical_features, pretrained=True):
        super(BrainHemorrhageModel, self).__init__()

        # 加载预训练的ResNet-50作为基础模型
        self.resnet = models.resnet50(pretrained=pretrained)

        # 修改ResNet的最后一层
        in_features = self.resnet.fc.in_features

        # 移除原始的全连接层
        self.resnet.fc = nn.Identity()

        # 创建新的分类器，结合ResNet特征和临床特征
        self.classifier = nn.Sequential(
            nn.Linear(in_features + num_clinical_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)  # 二分类输出
        )

    def forward(self, ct_image, clinical_features):
        # 通过ResNet提取CT图像特征
        img_features = self.resnet(ct_image)

        # 连接图像特征和临床特征
        combined_features = torch.cat((img_features, clinical_features), dim=1)

        # 通过分类器
        output = self.classifier(combined_features)

        return output

# 修改后的数据集类，增加了错误处理和日志
class BrainCTDataset(Dataset):
    def __init__(self, ct_paths, mask_paths, clinical_data, labels, transform=None):
        self.ct_paths = ct_paths
        self.mask_paths = mask_paths
        self.clinical_data = clinical_data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.ct_paths)

    def __getitem__(self, idx):
        try:
            # 加载CT图像和掩码
            ct_path = self.ct_paths[idx]
            mask_path = self.mask_paths[idx]

            # 检查文件是否存在
            if not os.path.exists(ct_path):
                raise FileNotFoundError(f"CT文件不存在: {ct_path}")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"掩码文件不存在: {mask_path}")

            # 使用nibabel读取NIfTI格式的CT图像数据
            ct_img = nib.load(ct_path)
            ct_data = ct_img.get_fdata()

            mask_img = nib.load(mask_path)
            mask_data = mask_img.get_fdata()
            mask_data = np.nan_to_num(mask_data)

            # 检查CT和掩码的形状是否匹配
            if ct_data.shape != mask_data.shape:
                print(f"警告: CT和掩码形状不匹配。CT: {ct_data.shape}, 掩码: {mask_data.shape}")
                # 尝试调整掩码大小以匹配CT
                if len(mask_data.shape) == len(ct_data.shape):
                    # 暂时保持现状，但记录警告信息
                    pass
                else:
                    raise ValueError(f"CT和掩码维度不匹配。CT: {len(ct_data.shape)}D, 掩码: {len(mask_data.shape)}D")

            # 安全获取中间切片（检查数组形状并防止越界）
            z_dim = ct_data.shape[2]
            if z_dim == 0:
                raise ValueError(f"CT数据的z维度为0: {ct_path}")

            def find_nonzero_slice(data):
                # Check which slices have any non-zero values
                nonzero_slices = []

                for i in range(data.shape[2]):  # Iterate through the 37 slices
                    slice_data = data[:, :, i]
                    if np.any(slice_data != 0):  # Check if the slice has any non-zero values
                        nonzero_slices.append(i)
                        # If you just want the first one, you can add:
                        # return i

                return nonzero_slices
            mid_slice_idx = find_nonzero_slice(mask_data)[0]
            ct_mid_slice = ct_data[:, :, mid_slice_idx]
            mask_mid_slice = mask_data[:, :, mid_slice_idx]

            # 检查切片是否为空或全为零
            if np.all(ct_mid_slice == 0):
                print(f"警告: CT中间切片全为零: {ct_path}")

            # 应用掩码前检查形状
            if ct_mid_slice.shape != mask_mid_slice.shape:
                print(f"警告: 切片形状不匹配。CT: {ct_mid_slice.shape}, 掩码: {mask_mid_slice.shape}")
                # 可以尝试调整，或者直接使用CT切片
                mask_mid_slice = np.ones_like(ct_mid_slice)  # 使用全1掩码作为回退

            # 应用掩码（安全地处理可能的值为零的情况）
            ct_masked = ct_mid_slice * mask_mid_slice

            # 安全标准化（处理可能的除零情况）
            ct_min = np.min(ct_masked)
            ct_max = np.max(ct_masked)
            if ct_max == ct_min:
                print(f"警告: CT数据没有值范围 (min=max={ct_min}): {ct_path}")
                ct_normalized = np.zeros_like(ct_masked)  # 如果没有值范围，返回全零数组
            else:
                ct_normalized = (ct_masked - ct_min) / (ct_max - ct_min + 1e-8)

            # 转换为3通道图像
            ct_3channel = np.stack([ct_normalized] * 3, axis=0)

            # 获取对应的临床数据
            clinical_features = self.clinical_data[idx]

            # 转换为Tensor
            ct_tensor = torch.FloatTensor(ct_3channel)
            clinical_tensor = torch.FloatTensor(clinical_features)
            label = torch.tensor(self.labels[idx], dtype=torch.long)

            # 应用数据增强
            if self.transform:
                try:
                    ct_tensor = self.transform(ct_tensor)
                except Exception as e:
                    print(f"数据增强错误: {str(e)}")
                    # 如果增强失败，使用原始tensor

            return ct_tensor, clinical_tensor, label

        except Exception as e:
            print(f"处理索引 {idx} 发生错误: {str(e)}")
            print(f"CT路径: {self.ct_paths[idx]}")
            print(f"掩码路径: {self.mask_paths[idx]}")

            # 创建一个默认返回值代替失败的样本
            default_shape = (3, 224, 224)  # 假设这是我们想要的默认形状
            default_ct = torch.zeros(default_shape)
            default_clinical = torch.zeros(len(self.clinical_data[0]))
            default_label = torch.tensor(0, dtype=torch.long)

            return default_ct, default_clinical, default_label

# 训练函数，增加错误处理
def train_model(model, dataloaders, criterion, optimizer, scheduler=None, num_epochs=25, device='cuda'):
    model = model.to(device)

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_auc': []
    }

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            all_labels = []
            all_probs = []
            samples_processed = 0

            # 启用错误处理的数据加载循环
            for batch_idx, data in enumerate(dataloaders[phase]):
                try:
                    ct_images, clinical_data, labels = data

                    # 检查批次数据是否有效
                    if ct_images.size(0) == 0:
                        print(f"警告: 批次 {batch_idx} 为空，跳过")
                        continue

                    ct_images = ct_images.to(device)
                    clinical_data = clinical_data.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(ct_images, clinical_data)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    batch_size = ct_images.size(0)
                    running_loss += loss.item() * batch_size
                    running_corrects += torch.sum(preds == labels.data)
                    samples_processed += batch_size

                    # 收集验证集的标签和概率，用于计算AUC
                    if phase == 'val':
                        all_labels.extend(labels.cpu().numpy())
                        probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
                        all_probs.extend(probs)

                except Exception as e:
                    print(f"处理批次 {batch_idx} 时出错: {str(e)}")
                    continue  # 跳过这个批次

            # 安全计算epoch指标
            if samples_processed > 0:
                epoch_loss = running_loss / samples_processed
                epoch_acc = running_corrects.double() / samples_processed

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # 记录训练历史
                if phase == 'train':
                    history['train_loss'].append(epoch_loss)
                    history['train_acc'].append(epoch_acc.cpu().numpy())
                else:
                    history['val_loss'].append(epoch_loss)
                    history['val_acc'].append(epoch_acc.cpu().numpy())

                    # 计算验证集的AUC (只有当有足够的样本时)
                    if len(all_labels) > 1 and len(set(all_labels)) > 1:
                        epoch_auc = roc_auc_score(all_labels, all_probs)
                        history['val_auc'].append(epoch_auc)
                        print(f'Validation AUC: {epoch_auc:.4f}')
                    else:
                        history['val_auc'].append(0.5)
                        print('警告: 无法计算AUC (需要两个不同的类别)')
            else:
                print(f"警告: 阶段 '{phase}' 没有处理任何样本!")
                if phase == 'train':
                    history['train_loss'].append(float('nan'))
                    history['train_acc'].append(float('nan'))
                else:
                    history['val_loss'].append(float('nan'))
                    history['val_acc'].append(float('nan'))
                    history['val_auc'].append(float('nan'))

        # 更新学习率
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(history['val_loss'][-1])
            else:
                scheduler.step()

        print()

    return model, history

def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for ct_images, clinical_data, labels in test_loader:
            ct_images = ct_images.to(device)
            clinical_data = clinical_data.to(device)

            outputs = model(ct_images, clinical_data)
            _, preds = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)[:, 1]

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate evaluation metrics
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds)

    # Safely calculate AUC (ensure there are two classes)
    if len(set(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs)
    else:
        auc = 0.5
        print("警告: 无法计算AUC，测试集中只有一个类别")

    return {
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'auc': auc,
        'predictions': all_preds,
        'probabilities': all_probs,
        'true_labels': all_labels
    }

def qc_drop(ct_paths, mask_paths, clinical_df, patient_id_column='patient_id'):
    """
    对CT扫描数据进行质量控制，剔除有问题的样本

    参数:
    ct_paths (list): CT文件路径列表
    mask_paths (list): 掩码文件路径列表
    clinical_df (pd.DataFrame): 包含临床数据的DataFrame
    patient_id_column (str): 临床数据中病人ID的列名

    返回:
    tuple: 过滤后的(ct_paths, mask_paths, clinical_df, valid_patient_ids)
    """
    import nibabel as nib
    import numpy as np
    import os

    # 确保输入列表长度一致
    assert len(ct_paths) == len(mask_paths), "CT路径和掩码路径数量不匹配"

    # 提取CT文件名中的病人ID
    def extract_patient_id(file_path):
        return os.path.basename(file_path).split('\\')[-1].split('.')[0].split('_')[0]

    # 初始化变量
    valid_indices = []
    valid_patient_ids = []

    print(f"开始质量控制，共 {len(ct_paths)} 个样本...")

    # 检查每个样本
    for i, (ct_path, mask_path) in enumerate(zip(ct_paths, mask_paths)):
        try:
            # 提取病人ID
            patient_id = extract_patient_id(ct_path)

            # 检查文件是否存在
            if not os.path.exists(ct_path):
                print(f"剔除: CT文件不存在: {ct_path}")
                continue
            if not os.path.exists(mask_path):
                print(f"剔除: 掩码文件不存在: {mask_path}")
                continue

            # 加载CT图像
            ct_img = nib.load(ct_path)
            ct_data = ct_img.get_fdata()

            # 检查CT维度
            if len(ct_data.shape) < 3:
                print(f"剔除: CT维度不足: {ct_path}, 形状: {ct_data.shape}")
                continue

            # 检查Z维度是否为0
            z_dim = ct_data.shape[2]
            if z_dim == 0:
                print(f"剔除: CT数据的z维度为0: {ct_path}")
                continue

            # 检查是否有值范围
            if np.min(ct_data) == np.max(ct_data):
                print(f"剔除: CT数据没有值范围 (min=max={np.min(ct_data)}): {ct_path}")
                continue

            # 检查是否包含无穷大或NaN值
            if np.isinf(ct_data).any(): # + np.isnan(ct_data).any()
                print(f"剔除: CT数据包含NaN或无穷值: {ct_path}")
                continue

            # 检查掩码
            try:
                mask_img = nib.load(mask_path)
                mask_data = mask_img.get_fdata()

                # 检查掩码是否为空
                if np.all(mask_data == 0):
                    print(f"剔除: 掩码全为零: {mask_path}")
                    continue

                if np.min(mask_data) == np.max(mask_data):
                    print(f"剔除: CT数据没有值范围 (min=max={np.min(mask_data)}): {mask_path}")
                    continue

                # 检查CT和掩码的形状是否匹配
                if ct_data.shape != mask_data.shape:
                    print(f"剔除: CT和掩码形状不匹配。CT: {ct_data.shape}, 掩码: {mask_data.shape}")
                    continue
            except Exception as e:
                print(f"剔除: 处理掩码文件出错: {str(e)}: {mask_path}")
                continue

            # 检查是否在临床数据中有对应记录
            if patient_id not in clinical_df[patient_id_column].values:
                print(f"剔除: 临床数据中未找到病人ID: {patient_id}")
                continue

            # 通过所有检查，添加到有效样本列表
            valid_indices.append(i)
            valid_patient_ids.append(patient_id)

        except Exception as e:
            print(f"剔除: 处理索引 {i} 发生错误: {str(e)}")
            print(f"CT路径: {ct_paths[i]}")
            print(f"掩码路径: {mask_paths[i]}")
            continue

    # 过滤CT和掩码路径
    filtered_ct_paths = [ct_paths[i] for i in valid_indices]
    filtered_mask_paths = [mask_paths[i] for i in valid_indices]

    # 过滤临床数据
    filtered_clinical_df = clinical_df[clinical_df[patient_id_column].isin(valid_patient_ids)]

    # 打印结果
    print(f"质量控制完成:")
    print(f"- 原始样本数: {len(ct_paths)}")
    print(f"- 有效样本数: {len(filtered_ct_paths)}")
    print(f"- 剔除样本数: {len(ct_paths) - len(filtered_ct_paths)}")
    print(f"- 有效临床数据: {filtered_clinical_df.shape[0]} 行")

    return filtered_ct_paths, filtered_mask_paths, filtered_clinical_df, valid_patient_ids





# 示例使用流程
def main():
    # 1. 准备数据路径和临床数据
    data_dir = "path/to/data"
    clinical_data_path = "path/to/clinical_data.csv"

    # 读取临床数据
    clinical_df = pd.read_csv(clinical_data_path)

    # 获取CT和mask文件路径
    ct_paths = [os.path.join(data_dir, "ct", f) for f in os.listdir(os.path.join(data_dir, "ct"))]
    mask_paths = [os.path.join(data_dir, "mask", f) for f in os.listdir(os.path.join(data_dir, "mask"))]

    # 确保文件顺序匹配
    ct_paths.sort()
    mask_paths.sort()

    # 处理临床数据
    # 假设clinical_df包含患者ID和对应的临床特征及标签
    patient_ids = [os.path.basename(p).split('.')[0] for p in ct_paths]

    # 提取对应患者的临床特征和标签
    clinical_features = []
    labels = []

    for patient_id in patient_ids:
        patient_data = clinical_df[clinical_df['patient_id'] == patient_id]
        if len(patient_data) > 0:
            # 提取需要的临床特征
            features = patient_data[['age', 'gender', 'feature1', 'feature2']].values[0]
            clinical_features.append(features)

            # 提取二分类标签
            label = patient_data['outcome'].values[0]
            labels.append(label)

    # 2. 划分训练集、验证集和测试集
    train_ct, test_ct, train_mask, test_mask, train_clinical, test_clinical, train_labels, test_labels = train_test_split(
        ct_paths, mask_paths, clinical_features, labels, test_size=0.2, random_state=42
    )

    train_ct, val_ct, train_mask, val_mask, train_clinical, val_clinical, train_labels, val_labels = train_test_split(
        train_ct, train_mask, train_clinical, train_labels, test_size=0.2, random_state=42
    )

    # 3. 数据增强
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 4. 创建数据集和数据加载器
    train_dataset = BrainCTDataset(train_ct, train_mask, train_clinical, train_labels, transform=train_transform)
    val_dataset = BrainCTDataset(val_ct, val_mask, val_clinical, val_labels, transform=val_transform)
    test_dataset = BrainCTDataset(test_ct, test_mask, test_clinical, test_labels, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }

    # 5. 初始化模型
    num_clinical_features = len(train_clinical[0])
    model = BrainHemorrhageModel(num_clinical_features=num_clinical_features)

    # 6. 设置训练参数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # 7. 训练模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model, history = train_model(
        model,
        dataloaders,
        criterion,
        optimizer,
        scheduler=scheduler,
        num_epochs=30,
        device=device
    )

    # 8. 评估模型
    results = evaluate_model(trained_model, test_loader, device=device)

    # 9. 可视化结果
    # 绘制训练历史
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.plot(history['val_auc'], label='Validation AUC')
    plt.title('Metrics Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 输出评估结果
    print("Confusion Matrix:")
    print(results['confusion_matrix'])
    print("\nClassification Report:")
    print(results['classification_report'])
    print(f"\nTest AUC: {results['auc']:.4f}")

if __name__ == "__main__":
    main()