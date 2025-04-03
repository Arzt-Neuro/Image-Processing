import os
import zipfile
import tempfile
import shutil
import xml.etree.ElementTree as ET
import numpy as np
import nibabel as nib
import glob
from pathlib import Path
import SimpleITK as sitk
from ipywidgets import interact, IntSlider
import matplotlib.pyplot as plt
import torch
from monai.transforms import Spacing
from monai.data import MetaTensor
import monai
import re

class MRBExtractor:
    """
    用于在标准Python环境中提取和解析3D Slicer MRB文件的类
    """

    def __init__(self, mrb_file_path):
        """
        初始化MRB提取器

        参数:
            mrb_file_path (str): MRB文件的路径
        """
        self.mrb_file_path = mrb_file_path
        self.scanid = mrb_file_path.split('\\')[-1].split('.')[0]
        self.temp_dir = None
        self.extracted_path = None
        self.scene_description = None
        self.volumes = {}
        self.segmentations = {}

    def extract(self):
        """
        将MRB文件解压到临时目录

        返回:
            bool: 提取是否成功
        """
        try:
            # 创建临时目录
            self.temp_dir = tempfile.mkdtemp()

            # 解压MRB文件 (MRB文件本质上是一个ZIP文件)
            with zipfile.ZipFile(self.mrb_file_path, 'r') as zip_ref:
                zip_ref.extractall(self.temp_dir)

            # MRB文件通常包含一个.mrml文件和一个Data目录
            self.extracted_path = self.temp_dir
            # 找到场景描述文件(.mrml)
            if len(glob.glob(self.extracted_path  + "/*.mrml")) == 0:
                self.extracted_path = self.extracted_path + '\\' + os.listdir(self.extracted_path)[0]

            mrml_files = list(Path(self.extracted_path).glob('*.mrml'))
            if not mrml_files:
                print(f"MRML-Datei im {self.mrb_file_path} wurde nicht gefunden.")
                return False

            # 解析场景描述文件
            self.scene_description = str(mrml_files[0])
            self._parse_scene()

            return True

        except Exception as e:
            print(f"Die Datei kann nicht extrahiert werden: {e}")
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
            self.temp_dir = None
            return False

    def _parse_scene(self):
        """
        根据提供的MRML文件结构解析场景文件，识别体积和分割数据
        """
        try:
            tree = ET.parse(self.scene_description)
            root = tree.getroot()

            print(f"########### Datei {self.scanid} wird analysiert ###########")

            # 1. 识别体积数据 - 直接查找Volume标签
            volume_nodes = root.findall(".//Volume")
            print(f"Es wurden {len(volume_nodes)} Knoten gefunden")

            for node in volume_nodes:
                node_id = node.get('id')
                name = node.get('name')

                # 查找存储节点引用
                storage_ref = None
                if 'references' in node.attrib:
                    refs = node.get('references').split(';')
                    for ref in refs:
                        if ref.startswith('storage:'):
                            storage_ref = ref.split(':')[1]
                            break

                if storage_ref:
                    print(f"Es wurden '{name}' gefunden (ID: {node_id}), Speicherreferenzen: {storage_ref}")
                    self.scan_ref = storage_ref

                    # 查找存储节点
                    storage_node = root.find(f".//*[@id='{storage_ref}']")
                    if storage_node:
                        file_name = storage_node.get('fileName')
                        if file_name:
                            # 修正文件路径
                            if not os.path.isabs(file_name):
                                file_path = os.path.join(self.extracted_path, file_name)
                            else:
                                file_path = file_name

                            if os.path.exists(file_path):
                                print(f"    Volumendaten vorhanden: {file_path}")
                                self.volumes[node_id] = {
                                    'name': name,
                                    'file_path': file_path
                                }
                            else:
                                print(f"Volumendaten sind nicht vorhanden: {file_path}")
                else:
                    print(f"    Volumendaten '{name}' habe Keine Referenzen gespeichert")

            # 2. 识别分割数据 - 直接查找Segmentation标签
            segmentation_nodes = root.findall(".//Segmentation")
            print(f"Es gibt {len(segmentation_nodes)} Segmentierung gefunden")

            for node in segmentation_nodes:
                node_id = node.get('id')
                name = node.get('name')

                # 查找存储节点引用
                storage_ref = None
                if 'references' in node.attrib:
                    refs = node.get('references').split(';')
                    for ref in refs:
                        if ref.startswith('storage:'):
                            storage_ref = ref.split(':')[1]
                            break

                if storage_ref:
                    print(f"Eine Segmentierung '{name}' wurde gefunden (ID: {node_id}), ihr Standort ist: {storage_ref}")
                    self.segm_ref = storage_ref

                    # 查找存储节点
                    storage_node = root.find(f".//*[@id='{storage_ref}']")
                    if storage_node:
                        file_name = storage_node.get('fileName')
                        if file_name:
                            # 修正文件路径
                            if not os.path.isabs(file_name):
                                file_path = os.path.join(self.extracted_path, file_name)
                            else:
                                file_path = file_name

                            if os.path.exists(file_path):
                                print(f"    Die Segmentierung befindet sich unter der Adresse: {file_path}")

                                # 查找参考体积
                                referenced_volume = None
                                if 'references' in node.attrib:
                                    refs = node.get('references').split(';')
                                    for ref in refs:
                                        if ref.startswith('referenceImageGeometryRef:'):
                                            referenced_volume = ref.split(':')[1]
                                            break

                                self.segmentations[node_id] = {
                                    'name': name,
                                    'file_path': file_path,
                                    'referenced_volume_node_id': referenced_volume
                                }
                            else:
                                print(f"    Die Segmentierung existiert nicht: {file_path}")
                else:
                    print(f"    Segmentierung '{name}' habe Keine Referenzen gespeichert")

            # 3. 如果上面的方法找不到数据，尝试直接查找存储节点
            if not self.volumes and not self.segmentations:
                print("\nVersuchen den Speicherknoten direkt zu finden...")

                # 查找体积存储
                volume_storage = root.findall(".//VolumeArchetypeStorage")
                for storage in volume_storage:
                    storage_id = storage.get('id')
                    file_name = storage.get('fileName')

                    if file_name:
                        # 修正文件路径
                        if not os.path.isabs(file_name):
                            file_path = os.path.join(self.extracted_path, file_name)
                        else:
                            file_path = file_name

                        if os.path.exists(file_path):
                            print(f"Volumendatei gefunden {storage_id}, Datei: {file_path}")

                            # 查找引用这个存储的体积节点
                            for node in root.findall(".//*[@references]"):
                                refs = node.get('references').split(';')
                                for ref in refs:
                                    if ref.startswith('storage:') and ref.split(':')[1] == storage_id:
                                        node_id = node.get('id')
                                        name = node.get('name')
                                        self.volumes[node_id] = {
                                            'name': name,
                                            'file_path': file_path
                                        }
                                        print(f"    Mit Volumen verbunden '{name}' (ID: {node_id})")

                # 查找分割存储
                segmentation_storage = root.findall(".//SegmentationStorage")
                for storage in segmentation_storage:
                    storage_id = storage.get('id')
                    file_name = storage.get('fileName')

                    if file_name:
                        # 修正文件路径
                        if not os.path.isabs(file_name):
                            file_path = os.path.join(self.extracted_path, file_name)
                        else:
                            file_path = file_name

                        if os.path.exists(file_path):
                            print(f"Segmentierungsdatei gefunden{storage_id}, Datei: {file_path}")

                            # 查找引用这个存储的分割节点
                            for node in root.findall(".//*[@references]"):
                                refs = node.get('references').split(';')
                                has_storage_ref = False
                                referenced_volume = None

                                for ref in refs:
                                    if ref.startswith('storage:') and ref.split(':')[1] == storage_id:
                                        has_storage_ref = True
                                    if ref.startswith('referenceImageGeometryRef:'):
                                        referenced_volume = ref.split(':')[1]

                                if has_storage_ref:
                                    node_id = node.get('id')
                                    name = node.get('name')
                                    self.segmentations[node_id] = {
                                        'name': name,
                                        'file_path': file_path,
                                        'referenced_volume_node_id': referenced_volume
                                    }
                                    print(f"    Es gibt '{name}' gefunden hat (ID: {node_id})")

            print(f"Analyse abgeschlossen, gibt es {len(self.volumes)} Volume und {len(self.segmentations)} Segmentierung")
            print('\n\n\n')

        except Exception as e:
            print(f"Fehler beim Parsen der Szene: {e}")
            import traceback
            traceback.print_exc()

    def get_volume(self):
        """
        获取所有体积数据的列表

        返回:
            dict: 体积ID到体积信息的映射
        """
        return self.volumes

    def get_segmentation(self):
        """
        获取所有分割数据的列表

        返回:
            dict: 分割ID到分割信息的映射
        """
        return self.segmentations

    def load_volume(self, volume_id=None):
        """
        加载特定ID的体积数据

        参数:
            volume_id (str): 要加载的体积的ID

        返回:
            tuple: (数据数组, 元数据) 如果加载成功，否则返回None
        """
        if volume_id is None:
            volume_id = list(self.volumes.keys())[0]
        if volume_id not in self.volumes:
            print(f"Die Datei für das Volumen mit der ID: {volume_id} nicht gefunden")
            return None

        volume_info = self.volumes[volume_id]
        file_path = volume_info['file_path']

        try:
            # 使用nibabel加载体积数据
            nib_img = nib.load(file_path)
            data = nib_img.get_fdata()
            meta = {
                'affine': nib_img.affine,
                'header': nib_img.header,
                'name': volume_info['name']
            }
            return data, meta
        except Exception as e1:
            try:
                # 尝试使用SimpleITK加载分割
                # 注: 需要安装SimpleITK: pip install SimpleITK

                img = sitk.ReadImage(file_path)
                data = sitk.GetArrayFromImage(img)
                meta = {
                    'origin': img.GetOrigin(),
                    'spacing': img.GetSpacing(),
                    'direction': img.GetDirection(),
                    'name': volume_info['name'],
                    'size': img.GetSize(),
                    'dimension': img.GetDimension(),
                    'pixel_type': img.GetPixelIDTypeAsString()
                }
                for key in img.GetMetaDataKeys():
                    try:
                        meta[key] = img.GetMetaData(key)
                    except Exception as e:
                        meta[key] = f"Error reading: {str(e)}"
                return data, meta
            except Exception as e2:
                print(f"Beim Lesen der Volumendaten ist ein Fehler aufgetreten: nibabel: {e1}, sitk: {e2}")
                return None

    def load_segmentation(self, segmentation_id=None):
        """
        加载特定ID的分割数据

        注意: 分割可能存储为.seg.nrrd文件，需要特殊处理
        这里我们用一个简化的方法，尝试使用SimpleITK读取

        参数:
            segmentation_id (str): 要加载的分割的ID

        返回:
            tuple: (数据数组, 元数据) 如果加载成功，否则返回None
        """
        if segmentation_id is None:
            segmentation_id = list(self.segmentations.keys())[0]
        if segmentation_id not in self.segmentations:
            print(f"Die Segmentierungsdaten für die ID: {segmentation_id} gefunden hat")
            return None

        segmentation_info = self.segmentations[segmentation_id]
        file_path = segmentation_info['file_path']

        try:
            # 尝试使用SimpleITK加载分割

            img = sitk.ReadImage(file_path)
            data = sitk.GetArrayFromImage(img)
            meta = {
                'origin': img.GetOrigin(),
                'spacing': img.GetSpacing(),
                'direction': img.GetDirection(),
                'name': segmentation_info['name'],
                'size': img.GetSize(),
                'dimension': img.GetDimension(),
                'pixel_type': img.GetPixelIDTypeAsString()
            }
            for key in img.GetMetaDataKeys():
                try:
                    meta[key] = img.GetMetaData(key)
                except Exception as e:
                    meta[key] = f"Error reading: {str(e)}"
            return data, meta
        except ImportError:
            print("无法导入SimpleITK，请使用 pip install SimpleITK 安装它")
            return None
        except Exception as e:
            print(f"    Fehler beim Parsen der Segmentierung: {e}")
            return None

    def close(self):
        """
        清理临时文件
        """
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None


# 使用示例
def main():
    # MRB文件路径
    mrb_file_path = "/path/to/your/file.mrb"

    # 创建提取器
    extractor = MRBExtractor(mrb_file_path)

    try:
        # 提取MRB文件
        if not extractor.extract():
            print("提取MRB文件失败")
            return

        # 获取体积列表
        volumes = extractor.get_volumes()
        print("\n体积:")
        for vol_id, vol_info in volumes.items():
            print(f"ID: {vol_id}, 名称: {vol_info['name']}, 文件: {vol_info['file_path']}")

        # 获取分割列表
        segmentations = extractor.get_segmentations()
        print("\n分割:")
        for seg_id, seg_info in segmentations.items():
            print(f"ID: {seg_id}, 名称: {seg_info['name']}, 文件: {seg_info['file_path']}")

        # 加载第一个体积(如果有)
        if volumes:
            first_vol_id = list(volumes.keys())[0]
            print(f"\n加载体积: {volumes[first_vol_id]['name']}")
            vol_data, vol_meta = extractor.load_volume(first_vol_id)
            if vol_data is not None:
                print(f"体积数据形状: {vol_data.shape}")
                print(f"体积数据类型: {vol_data.dtype}")
                print(f"元数据: {vol_meta}")

        # 加载第一个分割(如果有)
        if segmentations:
            first_seg_id = list(segmentations.keys())[0]
            print(f"\n加载分割: {segmentations[first_seg_id]['name']}")
            seg_data, seg_meta = extractor.load_segmentation(first_seg_id)
            if seg_data is not None:
                print(f"分割数据形状: {seg_data.shape}")
                print(f"分割数据类型: {seg_data.dtype}")
                print(f"元数据: {seg_meta}")

                # 基本统计
                unique_labels = np.unique(seg_data)
                print(f"唯一标签值: {unique_labels}")
                for label in unique_labels:
                    if label > 0:  # 跳过背景
                        count = np.sum(seg_data == label)
                        print(f"标签 {label} 的体素数量: {count}")

    finally:
        # 清理
        extractor.close()


if __name__ == "__main__":
    main()



# generate a nii object, with no function of SITK conversion
def convert_nii_vieux(mtx, meta, convert=True):
    direction_matrix = np.array(meta['direction']).reshape(3, 3)
    spacing_matrix = np.diag(meta['spacing'])
    origin = np.array(meta['origin'])

    # 构建4x4仿射矩阵
    affine = np.eye(4)
    affine[:3, :3] = direction_matrix @ spacing_matrix
    affine[:3, 3] = origin
    nifti_img = nib.Nifti1Image(mtx, affine)

    # 设置其他元数据
    header = nifti_img.header
    header.set_data_dtype(np.float32)

    return nifti_img

def convert_itk(mtx, meta):
    """
    从NumPy数组和元数据创建SimpleITK图像对象

    参数:
    - mtx: NumPy数组，包含图像数据
    - meta: 字典，包含图像元数据

    返回:
    - SimpleITK图像对象
    """
    # 从NumPy数组创建SimpleITK图像
    sitk_image = sitk.GetImageFromArray(mtx)

    # 设置基本元数据
    sitk_image.SetSpacing(meta['spacing'])
    sitk_image.SetOrigin(meta['origin'])

    # 处理方向矩阵
    # SimpleITK需要方向矩阵以特定格式：对于3D图像，是一个9元素的向量
    # 如果方向是一个3x3矩阵，我们需要将其展平为向量
    direction = np.array(meta['direction'])
    if direction.shape == (9,):  # 已经是向量形式
        sitk_image.SetDirection(direction)
    elif direction.shape == (3, 3):  # 3x3矩阵
        direction_vector = direction.flatten()
        sitk_image.SetDirection(direction_vector)

    # 设置其他元数据（可选）
    for key, value in meta.items():
        if key not in ['spacing', 'origin', 'direction', 'size', 'dimension']:
            # 尝试将元数据设置为字符串
            try:
                sitk_image.SetMetaData(key, str(value))
            except:
                # 如果设置失败，跳过该元数据
                print(f"Metadaten können nicht festgelegt werden: {key}")

    return sitk_image


# generate a nii object, with function of SITK conversion
def convert_nii(mtx, meta, convert=True):
    # 检查NRRD空间类型
    nrrd_space = meta.get('NRRD_space', '')

    # 处理矩阵轴顺序（NRRD和NIfTI可能有不同的轴顺序）
    # NRRD常见的是(z,y,x)顺序，而NIfTI期望的是(x,y,z)顺序
    if convert:
        # 对于常见的从NRRD到NIfTI的转换，通常需要转置并可能需要翻转某些轴
        mtx = np.transpose(mtx, (2, 1, 0))

        # 如果NRRD是LPS坐标系而NIfTI是RAS+坐标系，可能需要翻转某些轴
        if 'left-posterior-superior' in nrrd_space.lower():
            # 从LPS到RAS+转换（可能需要翻转x和y轴）
            mtx = np.flip(mtx, axis=(0, 1))

            # 需要调整方向矩阵和原点以反映这个变化
            direction_matrix = np.array(meta['direction']).reshape(3, 3)
            direction_matrix[:, 0] = -direction_matrix[:, 0]  # 翻转x方向
            direction_matrix[:, 1] = -direction_matrix[:, 1]  # 翻转y方向
            meta['direction'] = direction_matrix.flatten()

            # 同样需要调整原点
            origin = np.array(meta['origin'])
            # 计算图像边界的新原点（这是简化的，实际计算可能更复杂）
            size = np.array(meta['size'])
            spacing = np.array(meta['spacing'])
            origin[0] = origin[0] + (size[0] - 1) * spacing[0]
            origin[1] = origin[1] + (size[1] - 1) * spacing[1]
            meta['origin'] = origin

    # 构建4x4仿射矩阵
    direction_matrix = np.array(meta['direction']).reshape(3, 3)
    spacing_matrix = np.diag(meta['spacing'])
    origin = np.array(meta['origin'])

    affine = np.eye(4)
    affine[:3, :3] = direction_matrix @ spacing_matrix
    affine[:3, 3] = origin

    nifti_img = nib.Nifti1Image(mtx, affine)

    # 设置其他元数据
    header = nifti_img.header
    header.set_data_dtype(np.float32)

    # 设置坐标系统信息
    if convert and 'left-posterior-superior' in nrrd_space.lower():
        # 设置为RAS+坐标系
        header['qform_code'] = 1
        header['sform_code'] = 1

    return nifti_img

# read scans in a interactive fashion
# old version
def read_slice_vieux(data, axis=2):
    """使用ipywidgets创建可交互的切片查看器"""

    max_slice = data.shape[axis] - 1

    @interact(slice_num=IntSlider(min=0, max=max_slice, step=1, value=max_slice//2,
                                  description='Slice:',
                                  continuous_update=False))
    def _display_slice(slice_num):
        fig, ax = plt.subplots(figsize=(8, 8))

        if axis == 0:
            img = ax.imshow(data[slice_num, :, :], cmap='gray')
        elif axis == 1:
            img = ax.imshow(data[:, slice_num, :], cmap='gray')
        else:  # axis == 2
            img = ax.imshow(data[:, :, slice_num], cmap='gray')

        plt.colorbar(img, ax=ax)
        ax.set_title(f'Slice {slice_num}/{max_slice} along axis {axis}')
        plt.tight_layout()
        plt.show()


def read_slice(data, axis=2, level=40, window=80, mask=None):
    """使用ipywidgets创建可交互的切片查看器"""

    max_slice = data.shape[axis] - 1
    vmin = level - window/2
    vmax = level + window/2

    @interact(slice_num=IntSlider(min=0, max=max_slice, step=1, value=max_slice//2,
                                  description='Slice:',
                                  continuous_update=True))

    def _display_slice(slice_num):
        fig, ax = plt.subplots(figsize=(8, 8))

        if axis == 0:
            img_slice = data[slice_num, :, :]
            if mask is not None:
                mask_slice = mask[slice_num, :, :]
        elif axis == 1:
            img_slice = data[:, slice_num, :]
            if mask is not None:
                mask_slice = mask[:, slice_num, :]
        else:  # axis == 2
            img_slice = data[:, :, slice_num]
            if mask is not None:
                mask_slice = mask[:, :, slice_num]

        # 显示原始图像
        img = ax.imshow(img_slice, cmap='gray', vmin=vmin, vmax=vmax)
        mask_slice[mask_slice==0] = np.nan
        img_mask = ax.imshow(mask_slice, cmap='jet', alpha=0.3, vmin=0, vmax=1)
        plt.colorbar(img, ax=ax)
        ax.set_title(f'Slice {slice_num}/{max_slice} along axis {axis}')
        plt.tight_layout()
        plt.show()


# display single slice
# old version
def display_slice_vieux(data, n=None, axis=2):
    """显示单个切片"""
    if slice_num is None:
        slice_num = data.shape[axis] // 2

    if axis == 0:
        plt.imshow(data[slice_num, :, :], cmap='gray')
    elif axis == 1:
        plt.imshow(data[:, slice_num, :], cmap='gray')
    else:
        plt.imshow(data[:, :, slice_num], cmap='gray')

    plt.colorbar()
    plt.title(f'Slice {slice_num} along axis {axis}')

def display_slice(data, slice_num=None, axis=2, level=None, width=None):
    """
    显示单个切片，支持窗位窗宽调整

    参数:
    data - 3D数组
    slice_num - 要显示的切片索引，None则显示中间切片
    axis - 切片方向：0(矢状面)，1(冠状面)，2(轴向)
    window_level - 窗位（中心值），None则自动根据图像计算
    window_width - 窗宽，None则自动根据图像计算
    """
    if slice_num is None:
        slice_num = data.shape[axis] // 2

    # 获取当前切片数据
    if axis == 0:
        slice_data = data[slice_num, :, :]
    elif axis == 1:
        slice_data = data[:, slice_num, :]
    else:
        slice_data = data[:, :, slice_num]

    # 计算默认窗位窗宽（如果未指定）
    if level is None or width is None:
        # 使用当前切片的均值作为窗位
        auto_level = np.mean(slice_data)
        # 使用当前切片的标准差*2作为窗宽的一半
        auto_width = np.std(slice_data) * 4

        level = auto_level if level is None else level
        width = auto_width if width is None else width

    # 计算显示范围
    vmin = level - width/2
    vmax = level + width/2

    plt.figure(figsize=(10, 8))
    plt.imshow(slice_data, cmap='gray', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(f'Slice {slice_num} along axis {axis}\n'
              f'Window Level: {level:.1f}, Window Width: {width:.1f}')
    plt.tight_layout()


def resample_nii(nii_obj, thick=0.5, spatial_axis=2):
    """
    对NIfTI对象进行重采样以改变指定轴向上的层厚

    参数:
        nii_obj: nibabel的NIfTI对象
        thick: 目标层厚，默认0.5mm
        spatial_axis: 层厚所在的轴向索引，默认为2(通常Z轴)

    返回:
        重采样后的nibabel NIfTI对象
    """
    if isinstance(nii_obj, monai.data.meta_tensor.MetaTensor):
        monaimeta = nii_obj.meta
        nii_obj = nii_obj.numpy()
    else:
        monaimeta = None

    if isinstance(nii_obj, nib.nifti1.Nifti1Image):
        # 获取原始数据和元数据
        original_data = nii_obj.get_fdata()
        original_affine = nii_obj.affine
        original_header = nii_obj.header
        # 获取原始体素大小
        original_spacing = nii_obj.header.get_zooms()
        # 创建目标体素大小列表（只改变指定轴的间距）
        target_spacing = list(original_spacing)
        target_spacing[spatial_axis] = thick
        # 转换为PyTorch张量并添加批次维度
        data_tensor = torch.from_numpy(original_data).unsqueeze(0)

        # 创建MONAI的MetaTensor，并添加必要的元数据
        meta_dict = {
            'affine': torch.from_numpy(original_affine),
            'original_affine': torch.from_numpy(original_affine),
            'spatial_shape': torch.tensor(original_data.shape),
            'spacing': torch.tensor(original_spacing)
        }
        meta_data = MetaTensor(data_tensor, meta=meta_dict)
        # 创建Spacing转换
        spacing_transform = Spacing(
            pixdim=target_spacing,  # 目标体素大小
            mode="bilinear",  # 使用线性插值
            padding_mode="zeros"  # 边界填充方式
        )

        # 应用重采样变换
        resampled_data = spacing_transform(meta_data)
        # 转换回numpy数组并移除批次维度
        resampled_array = resampled_data.numpy()[0]
        # 获取新的仿射矩阵
        new_affine = resampled_data.meta['affine'].numpy()
        # 创建新的NIfTI对象
        new_nii = nib.Nifti1Image(resampled_array, new_affine, original_header)
        # 更新头信息中的体素尺寸
        new_nii.header.set_zooms(tuple(target_spacing))

    elif isinstance(nii_obj, np.ndarray):
        if nii_obj.ndim not in [3, 4]:
            raise ValueError("输入数组必须是3D [H,W,D]或4D [C,H,W,D]")

        # 处理3D输入，添加通道维度
        has_channel_dim = nii_obj.ndim == 4
        if not has_channel_dim:
            original_data = nii_obj[np.newaxis]  # 添加通道维度 [1,H,W,D]
        else:
            original_data = nii_obj

        # 假设默认体素间距为1.0mm
        original_spacing = [1.0, 1.0, 1.0]

        # 创建目标体素大小列表（只改变指定轴的间距）
        target_spacing = list(original_spacing)
        target_spacing[spatial_axis] = thick

        # 将numpy数组转换为PyTorch张量
        data_tensor = torch.from_numpy(original_data).float()

        # 为了使用MONAI的Spacing变换，需要创建一个伪仿射矩阵
        shape = original_data.shape[1:] if has_channel_dim else original_data.shape
        affine = np.eye(4)

        # 设置仿射矩阵中的间距信息
        for i, (orig, target) in enumerate(zip(original_spacing, target_spacing)):
            affine[i, i] = orig / target if orig != 0 and target != 0 else 1.0

        # 创建MONAI的MetaTensor
        meta_dict = {
            'affine': torch.from_numpy(affine.astype(np.float32)),
            'original_affine': torch.from_numpy(affine.astype(np.float32)),
            'spatial_shape': torch.tensor(shape),
            'spacing': torch.tensor(original_spacing)
        }
        meta_data = MetaTensor(data_tensor, meta=meta_dict)

        # 创建Spacing变换
        spacing_transform = Spacing(
            pixdim=target_spacing,
            mode="bilinear",
            padding_mode="zeros"
        )

        # 应用重采样变换
        resampled_data = spacing_transform(meta_data)

        # 转换回numpy数组
        resampled_array = resampled_data.numpy()

        # 如果原始输入是3D，去除通道维度
        if not has_channel_dim:
            resampled_array = resampled_array[0]

        new_nii = resampled_array
        if monaimeta is not None:
            torch_tensor = torch.from_numpy(new_nii)
            monaimeta['spatial_shape'] = np.array(new_nii.shape,  dtype=np.int16)
            new_nii = MetaTensor(torch_tensor, meta=monaimeta)

    else:
        raise TypeError('unsupported format')

    return new_nii

# Unnötige Maske löschen
def homo_maske(objekt, terms):
    pattern = r'Segment(\d+)_LabelValue'
    meta = objekt[1]
    mtx = objekt[0]
    segment_index = 0
    segment_mapping = {}

    # Continue until we don't find any more segments
    while any(re.compile(pattern).search(s) for s in list(meta.keys())):
        label_key = f'Segment{segment_index}_LabelValue'
        name_key = f'Segment{segment_index}_Name'
        try:
            segment_mapping[meta[name_key]] = meta[label_key]
            meta.pop(label_key)
            meta.pop(name_key)
        except:
            pass
        segment_index += 1
    label = [segment_mapping.get(schlussel) for schlussel in terms if schlussel in segment_mapping]
    mtx_neu = np.isin(mtx, label).astype(np.int8)
    return (mtx_neu, meta)