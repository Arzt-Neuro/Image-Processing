import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer

# 导入PyTorch和MONAI
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

# 导入MONAI组件
from monai.networks.nets import DenseNet121, resnet10
from monai.transforms import (
    LoadImaged,
    Compose,
    ScaleIntensityd,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
    ToTensord,
    Resized
)
from monai.metrics import ROCAUCMetric, ConfusionMatrixMetric
from monai.data import Dataset as MonaiDataset

# 用于特征提取的库
from monai.transforms import LoadImage
from monai.features import calculate_all_metrics

# 设置随机种子确保结果可重复
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

class HemorrhageExpansionPredictor:
    def __init__(self, device=None):
        self.clinical_features = None
        self.radiomics_features = None
        self.labels = None
        self.model = None
        self.history = None
        self.feature_names = None
        self.preprocessor = None
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

    def extract_radiomics_features(self, image_paths, mask_paths):
        """
        使用MONAI从CT图像和血肿分割掩膜中提取放射组学特征

        参数:
        image_paths: 原始CT图像路径列表
        mask_paths: 对应的血肿分割掩膜路径列表

        返回:
        radiomics_features: 包含所有放射组学特征的DataFrame
        """
        # 创建MONAI LoadImage转换
        load_image = LoadImage()

        features_list = []
        for img_path, mask_path in zip(image_paths, mask_paths):
            try:
                # 使用MONAI加载图像和掩膜
                image_obj, _ = load_image(img_path)
                mask_obj, _ = load_image(mask_path)

                # 将掩膜二值化
                mask_obj = torch.where(mask_obj > 0.5, torch.ones_like(mask_obj), torch.zeros_like(mask_obj))

                # 确保图像和掩膜在同一设备上
                image_obj = image_obj.to(self.device)
                mask_obj = mask_obj.to(self.device)

                # 使用MONAI计算特征
                # 不同特征类别
                feature_categories = [
                    "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm", "shape"
                ]

                feature_dict = {}

                # 对每个特征类别计算特征
                for category in feature_categories:
                    try:
                        # 使用MONAI的特征计算函数
                        category_features = calculate_all_metrics(
                            image=image_obj,
                            mask=mask_obj,
                            category=category
                        )
                        # 将特征添加到字典中
                        for key, value in category_features.items():
                            if isinstance(value, (int, float, np.number)):
                                feature_dict[f"{category}_{key}"] = float(value)
                            elif torch.is_tensor(value) and value.numel() == 1:
                                feature_dict[f"{category}_{key}"] = float(value.item())
                    except Exception as cat_err:
                        print(f"计算 {category} 特征时出错: {cat_err}")

                features_list.append(feature_dict)

            except Exception as e:
                print(f"处理 {img_path} 时出错: {e}")
                features_list.append({})  # 添加空字典以保持索引一致

        # 将特征列表转换为DataFrame
        radiomics_df = pd.DataFrame(features_list)

        # 填充可能的缺失值
        radiomics_df = radiomics_df.fillna(radiomics_df.mean())

        return radiomics_df

    def load_clinical_data(self, clinical_data_path):
        """
        加载临床数据

        参数:
        clinical_data_path: 临床数据CSV文件路径

        返回:
        clinical_data: 包含临床特征的DataFrame
        """
        clinical_data = pd.read_csv(clinical_data_path)
        return clinical_data

    def load_labels(self, labels_path):
        """
        加载血肿扩大标签数据

        参数:
        labels_path: 标签数据CSV文件路径

        返回:
        labels: 包含血肿扩大标签的Series (0=无扩大, 1=有扩大)
        """
        labels_df = pd.read_csv(labels_path)
        return labels_df['hematoma_expansion']

    def preprocess_data(self, clinical_data, radiomics_data):
        """
        预处理临床和影像组学数据

        参数:
        clinical_data: 临床数据DataFrame
        radiomics_data: 放射组学特征DataFrame

        返回:
        preprocessed_data: 预处理后的合并特征
        """
        # 识别数值和分类特征
        numerical_features = clinical_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = clinical_data.select_dtypes(include=['object', 'category']).columns.tolist()

        # 创建预处理器
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # 合并转换器
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # 预处理临床数据
        clinical_processed = self.preprocessor.fit_transform(clinical_data)

        # 标准化放射组学特征
        radiomics_scaler = StandardScaler()
        radiomics_processed = radiomics_scaler.fit_transform(radiomics_data)

        # 保存特征名称（用于后续特征重要性分析）
        self.feature_names = (
                numerical_features +
                list(self.preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)) +
                radiomics_data.columns.tolist()
        )

        # 合并预处理后的特征
        X_combined = np.hstack((clinical_processed, radiomics_processed))

        return X_combined

    def build_model(self, clinical_input_dim, radiomics_input_dim):
        """
        使用PyTorch和MONAI构建深度学习模型，包括两个输入分支和一个合并层

        参数:
        clinical_input_dim: 临床特征维度
        radiomics_input_dim: 放射组学特征维度

        返回:
        model: PyTorch模型
        """
        # 创建一个自定义的PyTorch模型
        class DualInputModel(nn.Module):
            def __init__(self, clinical_dim, radiomics_dim):
                super(DualInputModel, self).__init__()

                # 临床数据分支
                self.clinical_branch = nn.Sequential(
                    nn.Linear(clinical_dim, 64),
                    nn.ReLU(),
                    nn.BatchNorm1d(64),
                    nn.Dropout(0.3),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.BatchNorm1d(32)
                )

                # 放射组学特征分支
                self.radiomics_branch = nn.Sequential(
                    nn.Linear(radiomics_dim, 128),
                    nn.ReLU(),
                    nn.BatchNorm1d(128),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.BatchNorm1d(64)
                )

                # 合并后的层
                self.combined = nn.Sequential(
                    nn.Linear(32 + 64, 64),
                    nn.ReLU(),
                    nn.BatchNorm1d(64),
                    nn.Dropout(0.5),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.BatchNorm1d(32),
                    nn.Dropout(0.3),
                    nn.Linear(32, 1),
                    nn.Sigmoid()
                )

            def forward(self, clinical_input, radiomics_input):
                clinical_features = self.clinical_branch(clinical_input)
                radiomics_features = self.radiomics_branch(radiomics_input)

                # 连接特征
                combined = torch.cat((clinical_features, radiomics_features), dim=1)

                # 通过合并层
                output = self.combined(combined)
                return output

        # 创建模型并移动到适当的设备
        model = DualInputModel(clinical_input_dim, radiomics_input_dim).to(self.device)

        return model

    def train_model(self, X_clinical, X_radiomics, y, validation_split=0.2, epochs=100, batch_size=32):
        """
        使用PyTorch训练模型

        参数:
        X_clinical: 预处理后的临床特征
        X_radiomics: 预处理后的放射组学特征
        y: 血肿扩大标签
        validation_split: 验证集比例
        epochs: 训练轮数
        batch_size: 批次大小

        返回:
        history: 包含训练历史的字典
        """
        # 构建模型
        self.model = self.build_model(X_clinical.shape[1], X_radiomics.shape[1])

        # 分割训练集和验证集
        val_size = int(len(X_clinical) * validation_split)
        train_size = len(X_clinical) - val_size

        train_indices = list(range(train_size))
        val_indices = list(range(train_size, len(X_clinical)))

        # 转换为PyTorch张量
        X_clinical_tensor = torch.FloatTensor(X_clinical).to(self.device)
        X_radiomics_tensor = torch.FloatTensor(X_radiomics).to(self.device)
        y_tensor = torch.FloatTensor(y).view(-1, 1).to(self.device)

        # 创建训练和验证数据集
        train_clinical = X_clinical_tensor[train_indices]
        train_radiomics = X_radiomics_tensor[train_indices]
        train_labels = y_tensor[train_indices]

        val_clinical = X_clinical_tensor[val_indices]
        val_radiomics = X_radiomics_tensor[val_indices]
        val_labels = y_tensor[val_indices]

        # 创建数据加载器
        train_dataset = TensorDataset(train_clinical, train_radiomics, train_labels)
        val_dataset = TensorDataset(val_clinical, val_radiomics, val_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # 定义优化器和损失函数
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.BCELoss()

        # 训练历史记录
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_auc': [],
            'val_auc': []
        }

        # 指标
        roc_auc_metric = ROCAUCMetric()

        # 早停参数
        best_val_auc = 0
        patience = 20
        patience_counter = 0
        best_model_state = None

        # 训练循环
        for epoch in range(epochs):
            # 训练模式
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            train_predictions = []
            train_targets = []

            for clinical, radiomics, targets in train_loader:
                # 清除梯度
                optimizer.zero_grad()

                # 前向传播
                outputs = self.model(clinical, radiomics)

                # 计算损失
                loss = criterion(outputs, targets)

                # 反向传播
                loss.backward()

                # 更新权重
                optimizer.step()

                # 累积统计
                train_loss += loss.item()
                predicted = (outputs > 0.5).float()
                train_correct += (predicted == targets).sum().item()
                train_total += targets.size(0)

                # 收集用于AUC计算的预测值和目标值
                train_predictions.append(outputs.detach())
                train_targets.append(targets)

            # 计算训练AUC
            train_pred_concat = torch.cat(train_predictions, dim=0)
            train_target_concat = torch.cat(train_targets, dim=0)
            roc_auc_metric(train_pred_concat, train_target_concat)
            train_auc = roc_auc_metric.aggregate().item()
            roc_auc_metric.reset()

            # 验证模式
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            val_predictions = []
            val_targets = []

            with torch.no_grad():
                for clinical, radiomics, targets in val_loader:
                    # 前向传播
                    outputs = self.model(clinical, radiomics)

                    # 计算损失
                    loss = criterion(outputs, targets)

                    # 累积统计
                    val_loss += loss.item()
                    predicted = (outputs > 0.5).float()
                    val_correct += (predicted == targets).sum().item()
                    val_total += targets.size(0)

                    # 收集用于AUC计算的预测值和目标值
                    val_predictions.append(outputs)
                    val_targets.append(targets)

            # 计算验证AUC
            val_pred_concat = torch.cat(val_predictions, dim=0)
            val_target_concat = torch.cat(val_targets, dim=0)
            roc_auc_metric(val_pred_concat, val_target_concat)
            val_auc = roc_auc_metric.aggregate().item()
            roc_auc_metric.reset()

            # 计算平均指标
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total

            # 保存历史记录
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['train_auc'].append(train_auc)
            history['val_auc'].append(val_auc)

            # 打印进度
            print(f'Epoch {epoch+1}/{epochs} - '
                  f'Train loss: {avg_train_loss:.4f}, '
                  f'Val loss: {avg_val_loss:.4f}, '
                  f'Train acc: {train_acc:.4f}, '
                  f'Val acc: {val_acc:.4f}, '
                  f'Train AUC: {train_auc:.4f}, '
                  f'Val AUC: {val_auc:.4f}')

            # 早停检查
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'早停！在{epoch+1}轮后没有改善。')
                    break

        # 加载最佳模型
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        self.history = history
        return history

    def evaluate_model(self, X_clinical_test, X_radiomics_test, y_test):
        """
        评估模型性能

        参数:
        X_clinical_test: 测试集临床特征
        X_radiomics_test: 测试集放射组学特征
        y_test: 测试集标签

        返回:
        results: 包含各种性能指标的字典
        """
        # 预测测试集
        y_pred_prob = self.model.predict([X_clinical_test, X_radiomics_test])
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()

        # 计算ROC曲线和AUC
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        # 计算混淆矩阵
        cm = confusion_matrix(y_test, y_pred)

        # 计算分类报告（包括精确度、召回率、F1值）
        cr = classification_report(y_test, y_pred, output_dict=True)

        results = {
            'accuracy': cr['accuracy'],
            'sensitivity': cr['1']['recall'],  # 敏感度 = 召回率(类别1)
            'specificity': cr['0']['recall'],  # 特异度 = 召回率(类别0)
            'precision': cr['1']['precision'],
            'f1_score': cr['1']['f1-score'],
            'auc': roc_auc,
            'confusion_matrix': cm,
            'fpr': fpr,
            'tpr': tpr
        }

        return results

    def plot_results(self, results, history=None):
        """
        可视化模型性能

        参数:
        results: evaluate_model返回的结果字典
        history: 训练历史记录
        """
        plt.figure(figsize=(15, 10))

        # 1. ROC曲线
        plt.subplot(2, 2, 1)
        plt.plot(results['fpr'], results['tpr'], label=f'AUC = {results["auc"]:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()

        # 2. 混淆矩阵
        plt.subplot(2, 2, 2)
        sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Expansion', 'Expansion'],
                    yticklabels=['No Expansion', 'Expansion'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        if history:
            # 3. 准确率历史
            plt.subplot(2, 2, 3)
            plt.plot(history.history['accuracy'], label='Train')
            plt.plot(history.history['val_accuracy'], label='Validation')
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend()

            # 4. 损失历史
            plt.subplot(2, 2, 4)
            plt.plot(history.history['loss'], label='Train')
            plt.plot(history.history['val_loss'], label='Validation')
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend()

        plt.tight_layout()
        plt.savefig('model_performance.png', dpi=300)
        plt.show()

    def analyze_feature_importance(self, X_clinical, X_radiomics, y):
        """
        分析特征重要性

        参数:
        X_clinical: 预处理后的临床特征
        X_radiomics: 预处理后的放射组学特征
        y: 标签

        返回:
        feature_importance: 特征重要性DataFrame
        """
        from sklearn.ensemble import RandomForestClassifier

        # 合并特征
        X_combined = np.hstack((X_clinical, X_radiomics))

        # 使用随机森林评估特征重要性
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_combined, y)

        # 获取特征重要性
        importances = rf.feature_importances_

        # 创建特征重要性DataFrame
        feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        # 可视化前20个重要特征
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
        plt.title('Top 20 Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300)
        plt.show()

        return feature_importance

    def cross_validation(self, X_clinical, X_radiomics, y, n_splits=5, epochs=50, batch_size=32):
        """
        执行k折交叉验证

        参数:
        X_clinical: 预处理后的临床特征
        X_radiomics: 预处理后的放射组学特征
        y: 标签
        n_splits: 交叉验证折数
        epochs: 每折训练的轮数
        batch_size: 批次大小

        返回:
        cv_results: 包含每折性能指标的字典
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        cv_results = {
            'accuracy': [],
            'auc': [],
            'sensitivity': [],
            'specificity': []
        }

        for fold, (train_idx, test_idx) in enumerate(skf.split(X_clinical, y)):
            print(f"\nTraining fold {fold+1}/{n_splits}")

            # 分割数据
            X_clinical_train, X_clinical_test = X_clinical[train_idx], X_clinical[test_idx]
            X_radiomics_train, X_radiomics_test = X_radiomics[train_idx], X_radiomics[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # 构建模型
            model = self.build_model(X_clinical_train.shape[1], X_radiomics_train.shape[1])

            # 回调函数
            callbacks = [
                EarlyStopping(monitor='val_auc', patience=10, mode='max', restore_best_weights=True)
            ]

            # 训练模型
            history = model.fit(
                [X_clinical_train, X_radiomics_train], y_train,
                validation_split=0.2,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )

            # 评估模型
            y_pred_prob = model.predict([X_clinical_test, X_radiomics_test])
            y_pred = (y_pred_prob > 0.5).astype(int).flatten()

            # 计算ROC曲线和AUC
            fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
            roc_auc = auc(fpr, tpr)

            # 计算混淆矩阵
            cm = confusion_matrix(y_test, y_pred)

            # 计算分类报告
            cr = classification_report(y_test, y_pred, output_dict=True)

            # 保存结果
            cv_results['accuracy'].append(cr['accuracy'])
            cv_results['auc'].append(roc_auc)
            cv_results['sensitivity'].append(cr['1']['recall'])
            cv_results['specificity'].append(cr['0']['recall'])

            print(f"Fold {fold+1} - Accuracy: {cr['accuracy']:.4f}, AUC: {roc_auc:.4f}")

        # 计算平均性能
        for metric in cv_results:
            cv_results[f'mean_{metric}'] = np.mean(cv_results[metric])
            cv_results[f'std_{metric}'] = np.std(cv_results[metric])

        print("\nCross-Validation Results:")
        print(f"Mean Accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
        print(f"Mean AUC: {cv_results['mean_auc']:.4f} ± {cv_results['std_auc']:.4f}")
        print(f"Mean Sensitivity: {cv_results['mean_sensitivity']:.4f} ± {cv_results['std_sensitivity']:.4f}")
        print(f"Mean Specificity: {cv_results['mean_specificity']:.4f} ± {cv_results['std_specificity']:.4f}")

        return cv_results

    def save_model(self, model_path, preprocessor_path):
        """
        保存模型和预处理器

        参数:
        model_path: 模型保存路径
        preprocessor_path: 预处理器保存路径
        """
        # 保存模型
        self.model.save(model_path)

        # 保存预处理器
        import pickle
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(self.preprocessor, f)

        print(f"模型已保存至 {model_path}")
        print(f"预处理器已保存至 {preprocessor_path}")

    def load_model(self, model_path, preprocessor_path):
        """
        加载模型和预处理器

        参数:
        model_path: 模型加载路径
        preprocessor_path: 预处理器加载路径
        """
        # 加载模型
        self.model = tf.keras.models.load_model(model_path)

        # 加载预处理器
        import pickle
        with open(preprocessor_path, 'rb') as f:
            self.preprocessor = pickle.load(f)

        print(f"模型已加载自 {model_path}")
        print(f"预处理器已加载自 {preprocessor_path}")

    def predict_new_case(self, clinical_data, ct_image_path, mask_path):
        """
        预测新病例的血肿扩大风险

        参数:
        clinical_data: 包含临床特征的DataFrame (单行)
        ct_image_path: CT图像路径
        mask_path: 血肿分割掩膜路径

        返回:
        prediction: 血肿扩大的预测概率
        """
        # 提取放射组学特征
        radiomics_features = self.extract_radiomics_features([ct_image_path], [mask_path])

        # 预处理临床数据
        clinical_processed = self.preprocessor.transform(clinical_data)

        # 标准化放射组学特征
        radiomics_scaler = StandardScaler()
        radiomics_processed = radiomics_scaler.fit_transform(radiomics_features)

        # 预测
        prediction = self.model.predict([clinical_processed, radiomics_processed])

        # 返回预测概率
        return prediction[0][0]


# 示例用法
if __name__ == "__main__":
    # 初始化预测器
    predictor = HemorrhageExpansionPredictor()

    # 假设我们有以下数据路径
    clinical_data_path = 'clinical_data.csv'
    labels_path = 'hematoma_expansion_labels.csv'
    image_paths = ['patient1_ct.nii.gz', 'patient2_ct.nii.gz', ...]
    mask_paths = ['patient1_mask.nii.gz', 'patient2_mask.nii.gz', ...]

    # 1. 加载数据
    clinical_data = predictor.load_clinical_data(clinical_data_path)
    labels = predictor.load_labels(labels_path)

    # 2. 提取放射组学特征
    radiomics_features = predictor.extract_radiomics_features(image_paths, mask_paths)

    # 3. 预处理数据
    X_combined = predictor.preprocess_data(clinical_data, radiomics_features)

    # 4. 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # 假设前n列是临床特征，剩余列是放射组学特征
    n_clinical_features = len(clinical_data.columns)
    X_clinical_train, X_radiomics_train = X_train[:, :n_clinical_features], X_train[:, n_clinical_features:]
    X_clinical_test, X_radiomics_test = X_test[:, :n_clinical_features], X_test[:, n_clinical_features:]

    # 5. 训练模型
    history = predictor.train_model(
        X_clinical_train, X_radiomics_train, y_train,
        validation_split=0.2, epochs=100, batch_size=32
    )

    # 6. 评估模型
    results = predictor.evaluate_model(X_clinical_test, X_radiomics_test, y_test)

    # 7. 可视化结果
    predictor.plot_results(results, history)

    # 8. 分析特征重要性
    feature_importance = predictor.analyze_feature_importance(
        X_clinical_train, X_radiomics_train, y_train
    )

    # 9. 交叉验证
    cv_results = predictor.cross_validation(
        X_combined[:, :n_clinical_features],
        X_combined[:, n_clinical_features:],
        labels,
        n_splits=5
    )

    # 10. 保存模型
    predictor.save_model('hematoma_expansion_model.h5', 'preprocessor.pkl')