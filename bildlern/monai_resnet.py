import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import nibabel as nib
import monai
from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityd,
    ToTensord,
    Resized,
    RandRotated,
    RandZoomd,
    RandGaussianNoised,
)
from monai.networks.nets import DenseNet121, ResNet
from monai.metrics import ROCAUCMetric

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 自定义数据集类
class ICHDataset(Dataset):
    def __init__(self, ct_paths, mask_paths, clinical_data, labels, transform=None):
        """
        初始化脑出血CT数据集

        参数:
            ct_paths (list): CT图像路径列表
            mask_paths (list): CT mask路径列表
            clinical_data (pandas.DataFrame): 临床数据
            labels (list): 结局标签
            transform: 图像变换
        """
        self.ct_paths = ct_paths
        self.mask_paths = mask_paths
        self.clinical_data = clinical_data
        self.labels = labels
        self.transform = transform

        # 标准化临床数据
        self.scaler = StandardScaler()
        self.clinical_features = self.scaler.fit_transform(clinical_data.values)

    def __len__(self):
        return len(self.ct_paths)

    def __getitem__(self, idx):
        # 加载CT图像
        ct_img = nib.load(self.ct_paths[idx]).get_fdata()
        mask = nib.load(self.mask_paths[idx]).get_fdata()

        # 应用mask
        masked_ct = ct_img * mask

        # 获取临床特征
        clinical_features = torch.FloatTensor(self.clinical_features[idx])

        # 应用变换
        if self.transform:
            sample = {"image": masked_ct}
            sample = self.transform(sample)
            masked_ct = sample["image"]

        # 确保CT图像为4D张量 [C, D, H, W]
        if len(masked_ct.shape) == 3:
            masked_ct = masked_ct.unsqueeze(0)

        # 获取标签
        label = torch.FloatTensor([self.labels[idx]])

        return masked_ct, clinical_features, label

# 定义混合模型 (影像特征 + 临床特征)
class ICHNet(nn.Module):
    def __init__(self, clinical_features_dim, dropout_rate=0.5):
        super(ICHNet, self).__init__()

        # 3D CNN 用于CT影像特征提取 (使用MONAI的ResNet18)
        self.image_model = monai.networks.nets.ResNet(
            block="basic",
            layers=[2, 2, 2, 2],
            block_inplanes=[64, 128, 256, 512],
            spatial_dims=3,
            n_input_channels=1
        )

        # 获取CNN的输出特征维度
        self.image_features_dim = 512

        # 临床特征处理网络
        self.clinical_net = nn.Sequential(
            nn.Linear(clinical_features_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # 融合网络
        self.fusion_net = nn.Sequential(
            nn.Linear(self.image_features_dim + 32, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, images, clinical_features):
        # 提取影像特征
        image_features = self.image_model.forward(images)
        image_features = torch.mean(image_features, dim=[2, 3, 4])  # 全局平均池化

        # 处理临床特征
        clinical_features = self.clinical_net(clinical_features)

        # 特征融合
        combined_features = torch.cat((image_features, clinical_features), dim=1)

        # 最终预测
        output = self.fusion_net(combined_features)

        return output

# 数据加载和预处理
def prepare_dataloaders(
        ct_dir,
        mask_dir,
        clinical_data_path,
        batch_size=8,
        test_size=0.2,
        val_size=0.2
):
    """
    准备数据加载器

    参数:
        ct_dir (str): CT图像目录
        mask_dir (str): CT mask目录
        clinical_data_path (str): 临床数据CSV文件路径
        batch_size (int): 批次大小
        test_size (float): 测试集比例
        val_size (float): 验证集比例

    返回:
        tuple: (train_loader, val_loader, test_loader, clinical_features_dim)
    """
    # 加载临床数据
    clinical_df = pd.read_table(clinical_data_path, sep='\t')

    # 获取CT图像和mask文件路径
    ct_paths = sorted([os.path.join(ct_dir, f) for f in os.listdir(ct_dir) if f.endswith('.nii.gz')])
    mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.nii.gz')])

    # 确保数据对齐
    patient_ids = [os.path.basename(path).split('\\')[-1].split('.')[0].split('_')[0] for path in ct_paths]
    clinical_df = clinical_df[clinical_df['patient_id'].isin(patient_ids)]
    clinical_df = clinical_df.sort_values(by='patient_id')

    # 提取结局标签
    labels = clinical_df['outcome'].values  # 假设结局列名为'outcome'

    # 移除结局和ID列以获得临床特征
    clinical_features = clinical_df.drop(['patient_id', 'outcome'], axis=1)
    clinical_features_dim = clinical_features.shape[1]

    # 划分训练、验证和测试集
    train_val_ct, test_ct, train_val_mask, test_mask, train_val_clinical, test_clinical, train_val_labels, test_labels = train_test_split(
        ct_paths, mask_paths, clinical_features, labels, test_size=test_size, random_state=42, stratify=labels
    )

    train_ct, val_ct, train_mask, val_mask, train_clinical, val_clinical, train_labels, val_labels = train_test_split(
        train_val_ct, train_val_mask, train_val_clinical, train_val_labels,
        test_size=val_size/(1-test_size), random_state=42, stratify=train_val_labels
    )

    # 定义数据增强和预处理
    train_transforms = Compose([
        lambda x: {"image": np.expand_dims(x["image"], axis=0)},  # 添加通道维度
        ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),  # 标准化
        RandRotated(keys=["image"], range_x=15, range_y=15, range_z=15, prob=0.5),  # 随机旋转
        RandZoomd(keys=["image"], min_zoom=0.9, max_zoom=1.1, prob=0.5),  # 随机缩放
        RandGaussianNoised(keys=["image"], prob=0.5, mean=0.0, std=0.1),  # 随机噪声
        ToTensord(keys=["image"]),  # 转换为Tensor
    ])

    val_test_transforms = Compose([
        lambda x: {"image": np.expand_dims(x["image"], axis=0)},  # 添加通道维度
        ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),  # 标准化
        ToTensord(keys=["image"]),  # 转换为Tensor
    ])

    # 创建数据集
    train_dataset = ICHDataset(train_ct, train_mask, train_clinical, train_labels, transform=train_transforms)
    val_dataset = ICHDataset(val_ct, val_mask, val_clinical, val_labels, transform=val_test_transforms)
    test_dataset = ICHDataset(test_ct, test_mask, test_clinical, test_labels, transform=val_test_transforms)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader, clinical_features_dim

