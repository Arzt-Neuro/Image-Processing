import os
import numpy as np
import pandas as pd
import torch
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    Compose,
)
from monai.data import Dataset, DataLoader
import SimpleITK as sitk

def extract_all_radiomics(image_path, mask_path, output_path=None):
    """
    一键提取所有可用的影像组学特征

    参数:
        image_path: 影像文件路径
        mask_path: 掩码文件路径
        output_path: 可选的输出CSV路径

    返回:
        features_dict: 包含所有特征的字典
    """
    print(f"正在处理: {os.path.basename(image_path)}")

    # 1. 使用MONAI加载和预处理图像和掩码
    print("加载和预处理图像...")
    data = [{"image": image_path, "mask": mask_path}]

    transforms = Compose([
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]),
        Orientationd(keys=["image", "mask"], axcodes="RAS"),
        Spacingd(keys=["image", "mask"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=500, b_min=0, b_max=1, clip=True),
    ])

    dataset = Dataset(data=data, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=1)
    batch_data = next(iter(dataloader))

    # 2. 将处理后的数据转换为SimpleITK格式用于pyradiomics
    print("转换为SimpleITK格式...")
    processed_image = sitk.GetImageFromArray(batch_data["image"][0, 0].numpy())
    processed_mask = sitk.GetImageFromArray(batch_data["mask"][0, 0].numpy() > 0.5)

    # 设置SimpleITK图像的空间信息
    processed_image.SetSpacing([1.0, 1.0, 1.0])
    processed_mask.SetSpacing([1.0, 1.0, 1.0])

    # 3. 使用PyRadiomics提取所有特征
    print("提取所有组学特征...")
    import radiomics
    from radiomics import featureextractor

    # 设置PyRadiomics参数
    params = {}

    # 提取所有可用的特征
    extractor = featureextractor.RadiomicsFeatureExtractor(
        additionalInfo=True,
        # 启用所有特征类
        enableAllFeatures=True,
        # 启用所有图像类型
        enableAllImageTypes=True
    )

    # 实际提取特征
    print("开始特征提取...")
    try:
        features = extractor.execute(processed_image, processed_mask)
        print(f"成功提取 {len(features)} 个特征")
    except Exception as e:
        print(f"特征提取错误: {e}")
        return {"error": str(e)}

    # 4. 将SimpleITK特征转换为字典
    features_dict = {}
    for feature_name, feature_value in features.items():
        # 跳过诊断信息
        if feature_name.startswith('diagnostics_'):
            continue

        # 将数组转换为列表以便于保存
        if isinstance(feature_value, (np.ndarray, sitk.Image)):
            continue
        elif hasattr(feature_value, "GetSize"):
            continue

        # 添加到特征字典
        features_dict[feature_name] = feature_value

    # 5. 如果提供了输出路径，保存为CSV
    if output_path:
        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # 转换为DataFrame并保存
        df = pd.DataFrame([features_dict])
        df.to_csv(output_path, index=False)
        print(f"特征已保存至: {output_path}")

    return features_dict


def batch_extract_radiomics(image_dir, mask_dir, output_dir, file_extension=".nii.gz"):
    """
    批量处理文件夹中的所有影像，提取组学特征

    参数:
        image_dir: 包含影像文件的目录
        mask_dir: 包含掩码文件的目录
        output_dir: 输出目录
        file_extension: 文件扩展名，默认为.nii.gz
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有图像文件
    image_files = [f for f in os.listdir(image_dir) if f.endswith(file_extension)]
    print(f"找到 {len(image_files)} 个图像文件待处理")

    # 创建汇总结果DataFrame
    all_features = []

    # 处理每个文件
    for i, image_file in enumerate(image_files):
        print(f"\n处理 [{i+1}/{len(image_files)}]: {image_file}")

        # 从文件名获取ID
        case_id = os.path.splitext(image_file)[0]
        if file_extension == ".nii.gz":
            case_id = os.path.splitext(case_id)[0]  # 处理双扩展名

        # 查找对应的掩码文件
        mask_file = None
        for mf in os.listdir(mask_dir):
            if case_id in mf and mf.endswith(file_extension):
                mask_file = mf
                break

        if mask_file is None:
            print(f"警告: 未找到 {case_id} 的掩码文件，跳过")
            continue

        # 设置文件路径
        image_path = os.path.join(image_dir, image_file)
        mask_path = os.path.join(mask_dir, mask_file)
        output_path = os.path.join(output_dir, f"{case_id}_features.csv")

        # 提取特征
        try:
            features = extract_all_radiomics(image_path, mask_path, output_path)

            # 添加病例ID并收集结果
            features['case_id'] = case_id
            all_features.append(features)

            print(f"{case_id} 处理完成")
        except Exception as e:
            print(f"{case_id} 处理失败: {e}")

    # 保存汇总特征表
    if all_features:
        all_features_df = pd.DataFrame(all_features)
        summary_path = os.path.join(output_dir, "all_features.csv")
        all_features_df.to_csv(summary_path, index=False)
        print(f"\n汇总特征已保存至: {summary_path}")
        print(f"共提取 {len(all_features)} 个病例的特征，每个病例 {len(all_features[0]) - 1 if all_features else 0} 个特征")
    else:
        print("没有成功提取任何病例的特征")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='MONAI医学影像组学特征一键提取工具')
    parser.add_argument('--mode', type=str, choices=['single', 'batch'], default='single',
                        help='处理模式: single=单个文件, batch=批量处理')
    parser.add_argument('--image', type=str, help='影像文件路径或文件夹')
    parser.add_argument('--mask', type=str, help='掩码文件路径或文件夹')
    parser.add_argument('--output', type=str, help='输出文件路径或文件夹')
    parser.add_argument('--ext', type=str, default='.nii.gz', help='文件扩展名，批处理模式使用')

    args = parser.parse_args()

    if args.mode == 'single':
        if not all([args.image, args.mask]):
            print("错误: 单文件模式需要指定图像和掩码路径")
        else:
            extract_all_radiomics(args.image, args.mask, args.output)
    else:
        if not all([args.image, args.mask, args.output]):
            print("错误: 批处理模式需要指定图像文件夹、掩码文件夹和输出文件夹")
        else:
            batch_extract_radiomics(args.image, args.mask, args.output, args.ext)

# 使用示例:
# 单个文件:
# python extract_radiomics.py --mode single --image path/to/image.nii.gz --mask path/to/mask.nii.gz --output path/to/features.csv

# 批量处理:
# python extract_radiomics.py --mode batch --image path/to/images --mask path/to/masks --output path/to/results