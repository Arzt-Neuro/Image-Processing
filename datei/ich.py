from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from torch.utils.data import Dataset, Subset
import torch

from monai.config import DtypeLike
from monai.data.image_reader import ImageReader
from monai.transforms import LoadImage, Randomizable, apply_transform, Compose
from monai.utils import MAX_SEED, get_seed
import collections.abc


class ImageICH(Dataset, Randomizable):
    """
    脑出血血肿扩大预测专用数据集

    此数据集扩展了MONAI的ImageDataset，专门用于处理脑出血CT图像、血肿掩膜和扩大标签。
    支持与注意力机制DenseNet模型的无缝集成。

    Args:
        image_files: CT图像文件路径列表
        mask_files: 血肿掩膜文件路径列表
        labels: 血肿是否扩大的标签列表 (1表示扩大, 0表示不扩大)
        transform: 应用于CT图像的转换
        mask_transform: 应用于掩膜的转换
        label_transform: 应用于标签的转换
        image_only: 如果为True，仅返回图像数据；否则返回图像和元数据
        transform_with_metadata: 如果为True，元数据将传递给转换函数
        dtype: 加载图像的数据类型
        reader: 用于加载图像文件和元数据的读取器
    """

    def __init__(
            self,
            image_files: Sequence[str],
            seg_files: Sequence[str] | None = None,
            labels: Sequence[float] | None = None,
            clinical_features: Callable | None = None,
            transform: Callable | None = None,
            seg_transform: Callable | None = None,
            label_transform: Callable | None = None,
            image_only: bool = True,
            transform_with_metadata: bool = False,
            dtype: DtypeLike = np.float32,
            reader: ImageReader | str | None = None,
            return_dict: bool = False,
            *args,
            **kwargs,
    ) -> None:
        # 确保三个列表长度一致
        if seg_files is not None and (len(image_files) != len(seg_files)):
            raise ValueError(
                "Must have same the number of segmentation as image files: "
                f"images={len(image_files)}, segmentations={len(seg_files)}."
            )
        if labels is not None and (len(image_files) != len(labels)):
            raise ValueError(
                "Must have same the number of labels as image files: "
                f"images={len(image_files)}, labels={len(labels)}."
            )
        if clinical_features is not None and (len(image_files) != len(clinical_features)):
            raise ValueError(
                "Must have same the number of clinical_features as image files: "
                f"images={len(image_files)}, clinical_features={len(clinical_features)}."
            )

        self.image_files = image_files
        self.seg_files = seg_files
        self.labels = labels
        self.transform = transform
        self.seg_transform = seg_transform
        self.label_transform = label_transform
        if image_only and transform_with_metadata:
            raise ValueError("transform_with_metadata=True requires image_only=False.")
        self.image_only = image_only
        self.transform_with_metadata = transform_with_metadata
        self.loader = LoadImage(reader, image_only, dtype, *args, **kwargs)
        self.set_random_state(seed=get_seed())
        self._seed = 0  # transform synchronization seed
        self.return_dict = return_dict
        self.clinical_features = clinical_features

    def __len__(self) -> int:
        return len(self.image_files)

    def randomize(self, data: Any | None = None) -> None:
        self._seed = self.R.randint(MAX_SEED, dtype="uint32")

    def __getitem__(self, index: int):
        self.randomize()
        meta_data, seg_meta_data, seg, label, clinical_feature = None, None, None, None, None

        # load data and optionally meta
        if self.image_only:
            img = self.loader(self.image_files[index])
            if self.seg_files is not None:
                seg = self.loader(self.seg_files[index])
        else:
            img, meta_data = self.loader(self.image_files[index])
            if self.seg_files is not None:
                seg, seg_meta_data = self.loader(self.seg_files[index])

        # params for conventional machine vision
        hu_max = img.max()
        hu_min = img.min()
        csf_hu = (15 - hu_min) / (hu_max - hu_min)
        bone_hu = (120 - hu_min) / (hu_max - hu_min)

        # apply the transforms
        if self.transform is not None:
            if isinstance(self.transform, Randomizable):
                self.transform.set_random_state(seed=self._seed)

            if self.transform_with_metadata:
                img, meta_data = apply_transform(self.transform, (img, meta_data), map_items=False, unpack_items=True)
            else:
                img = apply_transform(self.transform, img, map_items=False)

        if self.seg_files is not None and self.seg_transform is not None:
            if isinstance(self.seg_transform, Randomizable):
                self.seg_transform.set_random_state(seed=self._seed)

            if self.transform_with_metadata:
                seg, seg_meta_data = apply_transform(
                    self.seg_transform, (seg, seg_meta_data), map_items=False, unpack_items=True
                )
            else:
                seg = apply_transform(self.seg_transform, seg, map_items=False)

        if self.labels is not None:
            label = self.labels[index]
            if self.label_transform is not None:
                label = apply_transform(self.label_transform, label, map_items=False)  # type: ignore
        if self.clinical_features is not None:
            clinical_feature = self.clinical_features[index]

        # conventional machine vision
        hu = img.clone()
        mask_hu = (hu < csf_hu) | (hu > bone_hu)
        hu[mask_hu] = 0

        # construct outputs
        if self.return_dict:
            data_dict = {
                'ct': img,
            }
            if seg is not None:
                data_dict['seg'] = seg
            data_dict['hu'] = hu
            if label is not None:
                data_dict['label'] = label
            if not self.image_only and meta_data is not None:
                data_dict['meta'] = meta_data
            if not self.image_only and seg_meta_data is not None:
                data_dict['meta_seg'] = seg_meta_data
            if clinical_feature is not None:
                data_dict['clinical'] = clinical_feature
            return data_dict
        else:
            data = [img]
            if seg is not None:
                data.append(seg)
            data.append(hu)
            if label is not None:
                data.append(label)
            if not self.image_only and meta_data is not None:
                data.append(meta_data)
            if not self.image_only and seg_meta_data is not None:
                data.append(seg_meta_data)
            if clinical_feature is not None:
                data.append(clinical_feature)
            if len(data) == 1:
                return data[0]
            # use tuple instead of list as the default collate_fn callback of MONAI DataLoader flattens nested lists
            return tuple(data)


class ImageDict(Dataset):
    """
    A generic dataset with a length property and an optional callable data transform
    when fetching a data sample.
    If passing slicing indices, will return a PyTorch Subset, for example: `data: Subset = dataset[1:4]`,
    for more details, please check: https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset

    For example, typical input data can be a list of dictionaries::

        [{                            {                            {
             'img': 'image1.nii.gz',      'img': 'image2.nii.gz',      'img': 'image3.nii.gz',
             'seg': 'label1.nii.gz',      'seg': 'label2.nii.gz',      'seg': 'label3.nii.gz',
             'extra': 123                 'extra': 456                 'extra': 789
         },                           },                           }]
    """

    def __init__(self,
                 data: Sequence,
                 transform: Sequence[Callable] | Callable | None = None,
                 labels: Sequence[float] | None = None,
                 clinical_features: Callable | None = None,
                 ) -> None:
        """
        Args:
            data: input data to load and transform to generate dataset for model.
            transform: a callable, sequence of callables or None. If transform is not
            a `Compose` instance, it will be wrapped in a `Compose` instance. Sequences
            of callables are applied in order and if `None` is passed, the data is returned as is.
        """
        self.data = data
        self.labels = labels
        self.clinical_features = clinical_features
        try:
            self.transform = Compose(transform) if not isinstance(transform, Compose) else transform
        except Exception as e:
            raise ValueError("`transform` must be a callable or a list of callables that is Composable") from e

    def __len__(self) -> int:
        return len(self.data)

    def _transform(self, index: int):
        """
        Fetch single data item from `self.data`.
        """
        data_i = self.data[index]
        return self.transform(data_i)

    def __getitem__(self, index: int | slice | Sequence[int]):
        """
        Returns a `Subset` if `index` is a slice or Sequence, a data item otherwise.
        """
        if isinstance(index, slice):
            # dataset[:42]
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return Subset(dataset=self, indices=indices)
        if isinstance(index, collections.abc.Sequence):
            # dataset[[1, 3, 4]]
            return Subset(dataset=self, indices=index)
        return self._transform(index)
