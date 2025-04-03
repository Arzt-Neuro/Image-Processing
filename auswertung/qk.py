import monai
import torch
import nibabel as nib
import numpy as np
import os
from tqdm.notebook import tqdm
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from monai.transforms import (
    LoadImage,
    ScaleIntensity,
    EnsureChannelFirst,
    Resize,
    Compose,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def oeffnen(image_path):
    transform = Compose([
        LoadImage(image_only=True, reader="NibabelReader"),
        ScaleIntensity(),
        EnsureChannelFirst(channel_dim=None),
        Resize((512, 512, 32))
    ])
    image = monai.data.Dataset([image_path], transform=transform)[0].numpy()
    return image

def qk_maske(data_df, scan_path_col, mask_path_col, patient_id_col='patient_id'):
    """
    对CT扫描数据进行质量控制，剔除有问题的样本

    参数:
    ct_paths (list): CT文件路径列表
    mask_paths (list): 掩码文件路径列表
    data_df (pd.DataFrame): 包含临床数据的DataFrame
    patient_id (str): 临床数据中病人ID的列名

    返回:
    tuple: 过滤后的(ct_paths, mask_paths, data_df, valid_patient_ids)
    """
    valid_indices = []
    valid_patient_ids = []

    for idx in tqdm(data_df.index):
        patient_id = data_df.loc[idx, patient_id_col]
        scan_path = data_df.loc[idx, scan_path_col]
        mask_path = data_df.loc[idx, mask_path_col]
        try:
            # 检查文件是否存在
            if not os.path.exists(scan_path):
                print(f"剔除: CT文件不存在: {scan_path}")
                continue
            if not os.path.exists(mask_path):
                print(f"剔除: 掩码文件不存在: {mask_path}")
                continue

            scan_data = oeffnen(scan_path)

            # 检查CT维度
            if len(scan_data.shape) < 3:
                print(f"剔除: CT维度不足: {patient_id}, 形状: {scan_data.shape}")
                continue

            # 检查Z维度是否为0
            z_dim = scan_data.shape[2]
            if z_dim == 0:
                print(f"剔除: CT数据的z维度为0: {patient_id}")
                continue

            # 检查是否有值范围
            if np.min(scan_data) == np.max(scan_data):
                print(f"剔除: CT数据没有值范围 (min=max={np.min(scan_data)}): {patient_id}")
                continue

            # 检查是否包含无穷大或NaN值
            if np.isinf(scan_data).any(): # + np.isnan(scan_data).any()
                print(f"剔除: CT数据包含NaN或无穷值: {patient_id}")
                continue

            # 增加：检查是否为全零或接近零的数据
            if abs(np.max(scan_data)) < 1e-5:
                print(f"剔除: CT数据全为接近零的值: {patient_id}")
                continue

            # 检查掩码
            try:
                mask_data = oeffnen(mask_path)

                # 检查掩码是否为空
                if np.all(mask_data == 0):
                    print(f"剔除: 掩码全为零: {patient_id}")
                    continue

                if np.min(mask_data) == np.max(mask_data):
                    print(f"剔除: CT数据没有值范围 (min=max={np.min(mask_data)}): {patient_id}")
                    continue
                def nehmen_slice_idx(data):
                    # Check which slices have any non-zero values
                    area = []
                    for i in range(data.shape[2]):
                        area.append(data[:, :, i].sum())
                    idx = int(np.argmax(area))
                    return idx
                scheibensequenz = nehmen_slice_idx(mask_data)
                ct_mid_slice = scan_data[:, :, scheibensequenz]
                mask_mid_slice = mask_data[:, :, scheibensequenz]

                # 应用掩码（安全地处理可能的值为零的情况）
                ct_masked = ct_mid_slice * mask_mid_slice

                # 安全标准化（处理可能的除零情况）
                ct_min = np.min(ct_masked)
                ct_max = np.max(ct_masked)
                if ct_max == ct_min:
                    print(f"警告: CT数据没有值范围 (min=max={ct_min}): {patient_id}, slice: {scheibensequenz}")
                    continue

                # 检查CT和掩码的形状是否匹配
                if scan_data.shape != mask_data.shape:
                    print(f"剔除: CT和掩码形状不匹配。CT: {scan_data.shape}, 掩码: {mask_data.shape}")
                    continue
            except Exception as e:
                print(f"剔除: 处理掩码文件出错: {str(e)}: {patient_id}")
                continue

            # 通过所有检查，添加到有效样本列表
            valid_indices.append(idx)
            valid_patient_ids.append(patient_id)

        except Exception as e:
            print(f"剔除: 处理索引 {idx} - {patient_id} 发生错误: {str(e)}")
            continue
    # nehmen der neuer tabelle
    filtered_data_df = data_df[data_df[patient_id_col].isin(valid_patient_ids)]

    # QK-Ergebnisse drucken
    print(f"Qualitätskontrolle fertig:")
    print(f"- Ursprüngliche Anzahl der Proben: {data_df.shape[0]}")
    print(f"- Anzahl gültiger Proben: {len(valid_indices)}")
    print(f"- Anzahl der entnommenen Proben: {data_df.shape[0] - len(valid_indices)}")
    return filtered_data_df, valid_indices, valid_patient_ids



def qk_scan(data_df, scan_path_col, patient_id_col='patient_id'):
    """
    对CT扫描数据进行质量控制，剔除有问题的样本

    参数:
    ct_paths (list): CT文件路径列表
    mask_paths (list): 掩码文件路径列表
    data_df (pd.DataFrame): 包含临床数据的DataFrame
    patient_id (str): 临床数据中病人ID的列名

    返回:
    tuple: 过滤后的(ct_paths, mask_paths, data_df, valid_patient_ids)
    """
    valid_indices = []
    valid_patient_ids = []

    for idx in tqdm(data_df.index):
        patient_id = data_df.loc[idx, patient_id_col]
        scan_path = data_df.loc[idx, scan_path_col]
        try:
            # 检查文件是否存在
            if not os.path.exists(scan_path):
                print(f"剔除: CT文件不存在: {scan_path}")
                continue

            scan_data = oeffnen(scan_path)

            # 检查CT维度
            if len(scan_data.shape) < 3:
                print(f"剔除: CT维度不足: {patient_id}, 形状: {scan_data.shape}")
                continue

            # 检查Z维度是否为0
            z_dim = scan_data.shape[2]
            if z_dim == 0:
                print(f"剔除: CT数据的z维度为0: {patient_id}")
                continue

            # 检查是否有值范围
            if np.min(scan_data) == np.max(scan_data):
                print(f"剔除: CT数据没有值范围 (min=max={np.min(scan_data)}): {patient_id}")
                continue

            # 检查是否包含无穷大或NaN值
            if np.isinf(scan_data).any(): # + np.isnan(scan_data).any()
                print(f"剔除: CT数据包含NaN或无穷值: {patient_id}")
                continue

            # 增加：检查是否为全零或接近零的数据
            if abs(np.max(scan_data)) < 1e-5:
                print(f"剔除: CT数据全为接近零的值: {patient_id}")
                continue

            ct_min = np.min(scan_data)
            ct_max = np.max(scan_data)
            if ct_max == ct_min:
                print(f"警告: CT数据没有值范围 (min=max={ct_min}): {patient_id}")
                continue

            # 通过所有检查，添加到有效样本列表
            valid_indices.append(idx)
            valid_patient_ids.append(patient_id)

        except Exception as e:
            print(f"剔除: 处理索引 {idx} - {patient_id} 发生错误: {str(e)}")
            continue
    # nehmen der neuer tabelle
    filtered_data_df = data_df[data_df[patient_id_col].isin(valid_patient_ids)]

    # QK-Ergebnisse drucken
    print(f"Qualitätskontrolle fertig:")
    print(f"- Ursprüngliche Anzahl der Proben: {data_df.shape[0]}")
    print(f"- Anzahl gültiger Proben: {len(valid_indices)}")
    print(f"- Anzahl der entnommenen Proben: {data_df.shape[0] - len(valid_indices)}")
    return filtered_data_df, valid_indices, valid_patient_ids

def qk_einzelne(patient_id, scan_path, mask_path=None, idx=None):
    try:
        # 检查文件是否存在
        if not os.path.exists(scan_path):
            print(f"剔除: CT文件不存在: {scan_path}")
            return patient_id, False, idx

        scan_data = oeffnen(scan_path)

        # 检查CT维度
        if len(scan_data.shape) < 3:
            print(f"剔除: CT维度不足: {patient_id}, 形状: {scan_data.shape}")
            return patient_id, False, idx

        # 检查Z维度是否为0
        z_dim = scan_data.shape[2]
        if z_dim == 0:
            print(f"剔除: CT数据的z维度为0: {patient_id}")
            return patient_id, False, idx

        # 检查是否有值范围
        if np.min(scan_data) == np.max(scan_data):
            print(f"剔除: CT数据没有值范围 (min=max={np.min(scan_data)}): {patient_id}")
            return patient_id, False, idx

        # 检查是否包含无穷大或NaN值
        if np.isinf(scan_data).any(): # + np.isnan(scan_data).any()
            print(f"剔除: CT数据包含NaN或无穷值: {patient_id}")
            return patient_id, False, idx

        # 增加：检查是否为全零或接近零的数据
        if abs(np.max(scan_data)) < 1e-5:
            print(f"剔除: CT数据全为接近零的值: {patient_id}")
            return patient_id, False, idx

        # 检查掩码
        if mask_path is not None:
            try:
                mask_data = oeffnen(mask_path)

                # 检查掩码是否为空
                if np.all(mask_data == 0):
                    print(f"剔除: 掩码全为零: {patient_id}")
                    return patient_id, False, idx

                if np.min(mask_data) == np.max(mask_data):
                    print(f"剔除: CT数据没有值范围 (min=max={np.min(mask_data)}): {patient_id}")
                    return patient_id, False, idx
                def nehmen_slice_idx(data):
                    # Check which slices have any non-zero values
                    area = []
                    for i in range(data.shape[2]):
                        area.append(data[:, :, i].sum())
                    idx = int(np.argmax(area))
                    return idx
                scheibensequenz = nehmen_slice_idx(mask_data)
                ct_mid_slice = scan_data[:, :, scheibensequenz]
                mask_mid_slice = mask_data[:, :, scheibensequenz]

                # 应用掩码（安全地处理可能的值为零的情况）
                ct_masked = ct_mid_slice * mask_mid_slice

                # 安全标准化（处理可能的除零情况）
                ct_min = np.min(ct_masked)
                ct_max = np.max(ct_masked)
                if ct_max == ct_min:
                    print(f"警告: CT数据没有值范围 (min=max={ct_min}): {patient_id}, slice: {scheibensequenz}")
                    return patient_id, False, idx

                # 检查CT和掩码的形状是否匹配
                if scan_data.shape != mask_data.shape:
                    print(f"剔除: CT和掩码形状不匹配。CT: {scan_data.shape}, 掩码: {mask_data.shape}")
                    return patient_id, False, idx
            except Exception as e:
                print(f"剔除: 处理掩码文件出错: {str(e)}: {patient_id}")
                return patient_id, False, idx

        # 通过所有检查，添加到有效样本列表
        return patient_id, True, idx

    except Exception as e:
        print(f"剔除: 处理索引 - {patient_id} 发生错误: {str(e)}")
        return patient_id, False, idx

def qk_multi(data_df, scan_path_col, mask_path_col=None, patient_id_col='patient_id', n_jobs=None):
    """
    对CT扫描数据进行并行质量控制，剔除有问题的样本

    参数:
    data_df (pd.DataFrame): 包含临床数据的DataFrame
    scan_path_col (str): 扫描路径列名
    patient_id_col (str): 临床数据中病人ID的列名
    n_jobs (int): 使用的CPU核心数，默认为None表示使用全部可用核心数减1

    返回:
    tuple: 过滤后的(filtered_data_df, valid_indices, valid_patient_ids)
    """
    if n_jobs is None:
        n_jobs = max(1, multiprocessing.cpu_count() - 1)

    print(f"Verwenden Sie {n_jobs} CPU-Kerne für die parallele Verarbeitung …")

    valid_indices = []
    valid_patient_ids = []

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = []
        for idx in data_df.index:
            patient_id = data_df.loc[idx, patient_id_col]
            scan_path = data_df.loc[idx, scan_path_col]
            if mask_path_col is not None:
                mask_path = data_df.loc[idx, mask_path_col]
            else:
                mask_path = None
            futures.append(executor.submit(qk_einzelne, patient_id, scan_path, mask_path, idx))

        for future in tqdm(as_completed(futures), total=len(futures), desc="CT-Scan wird verarbeitet"):
            try:
                patient_id, is_valid, idx = future.result()
                if is_valid:
                    valid_indices.append(idx)
                    valid_patient_ids.append(patient_id)
            except Exception as e:
                print(f"处理任务时发生错误: {str(e)}")

    filtered_data_df = data_df.loc[valid_indices,:]
    print(f"Qualitätskontrolle fertig:")
    print(f"- Ursprüngliche Anzahl der Proben: {data_df.shape[0]}")
    print(f"- Anzahl gültiger Proben: {len(valid_indices)}")
    print(f"- Anzahl der entnommenen Proben: {data_df.shape[0] - len(valid_indices)}")
    return filtered_data_df, valid_indices, valid_patient_ids

# example
"""
# 导入必要的库
import pandas as pd

# 准备数据
data_df = pd.DataFrame({
    'patient_id': ['P001', 'P002', 'P003', ...],
    'scan_path': ['/path/to/scan1.nii', '/path/to/scan2.nii', ...],
    # 其他列...
})

# 运行并行质量控制
filtered_df, valid_indices, valid_patient_ids = qk_scan_parallel(
    data_df=data_df,
    scan_path_col='scan_path',
    patient_id_col='patient_id',
    n_jobs=4  # 使用4个核心，可以根据您的系统调整
)
"""

def main():
    n = 100
    patient_ids = [f"PAT{i:04d}" for i in range(1, n+1)]
    scan_paths = [f"/data/scans/scan_{i:04d}.nii.gz" for i in range(1, n+1)]
    mask_paths = [f"/data/masks/mask_{i:04d}.nii.gz" for i in range(1, n+1)]
    armaturenbrett = pd.DataFrame({
        'patient_id': patient_ids,
        'scan_addr': scan_paths,
        'mask_addr': mask_paths,
        'gender': np.random.choice(['M', 'F'], size=n),
        'age': np.random.randint(18, 90, size=n),
        'weight': np.random.normal(70, 15, size=n).round(1),
        'height': np.random.normal(170, 15, size=n).round(1),
        'modality': np.random.choice(['CT', 'MRI', 'PET'], size=n),
        'date': pd.date_range(start='2023-01-01', periods=n)
    })

    oeffnen('test\\full_address\\test.nii.gz')
    qk_maske(data_df=armaturenbrett, scan_path_col='scan_addr', mask_path_col='mask_addr', patient_id_col='patient_id')
    qk_scan(data_df=armaturenbrett, scan_path_col='scan_addr',patient_id_col='patient_id')

if __name__ == "__main__":
    main()