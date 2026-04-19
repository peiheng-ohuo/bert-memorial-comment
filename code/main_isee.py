### import ###
import torch
import pickle
import random
import argparse
import itertools
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from nltk import accuracy
import os

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import  f1_score, accuracy_score, precision_score, recall_score, \
    roc_auc_score, confusion_matrix
from skmultilearn.model_selection import IterativeStratification
from transformers import AutoModel, AutoTokenizer, AutoConfig
from lime.lime_text import LimeTextExplainer
import shap

import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict, Counter



# random_seed
random_seed = 42 # just for reproduce
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)



# parser
parser = argparse.ArgumentParser()

parser.add_argument('--raw_data', type=str, default='data/both_isee_allemperor/both_isee_allemperor.xlsx')
parser.add_argument('--pkl_data', type=str, default='data/both_isee_allemperor/both_isee_allemperor.pkl')
parser.add_argument('--train_dataset', type=str, default='data/both_isee_allemperor/both_isee_allemperor_train.pt')
parser.add_argument('--valid_dataset', type=str, default='data/both_isee_allemperor/both_isee_allemperor_valid.pt')
parser.add_argument('--test_dataset', type=str, default='data/both_isee_allemperor/both_isee_allemperor_test.pt')
parser.add_argument('--infer_file', type=str, default='data/both_isee_allemperor/both_isee_allemperor_test_infer.xlsx')
parser.add_argument('--memorial_data', type=str, default='data/both_isee_allemperor/both_allemperor.xlsx')
parser.add_argument('--full_dataset', type=str, default='data/both_isee_allemperor/both_allemperor.pt')
parser.add_argument('--output_dataset', type=str, default='data/both_isee_allemperor/both_isee_allemperor_predict.xlsx')

parser.add_argument('--process_data', action='store_true')
parser.add_argument('--check_dataset', action='store_true')
parser.add_argument('--train', action='store_true')
parser.add_argument('--check_test_dataset', action='store_true')
parser.add_argument('--infer_test_dataset', action='store_true')
parser.add_argument('--infer_full_dataset', action='store_true')

parser.add_argument('--train_ratio', type=float, default=0.7)
parser.add_argument('--valid_ratio', type=float, default=0.2)
parser.add_argument('--num_epochs', type=int, default=6)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_classes', type=int, default=1)

parser.add_argument('--save_autobest_checkpoint', type=str, default='checkpoints/both_isee_allemperor_12.23_autobest.pt')
parser.add_argument('--save_epoch_checkpoint', type=str, default='checkpoints/both_isee_allemperor_12.23_epoch{}.pt')
parser.add_argument('--best_model', type=str, default='best_model/both_isee_allemperor_12.23_epoch3.pt') # 修改为选择的最优模型
parser.add_argument('--tensorboard', type=str, default='logs')

args = parser.parse_args()



# CustomDataset
class CustomDataset(Dataset):
    def __init__(self, data):
        self.texts = data['memorial'].tolist()
        self.labels = data.iloc[:, 1].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return text, label


# process_data
def process_data():
    print("开始处理数据...")
    data = pd.read_excel(args.raw_data)

    assert data.shape[1] == 2, "数据必须只有两列"
    text_col = data.columns[0]
    label_col = data.columns[1]
    print(f"正样本比例: {data[label_col].mean():.2%}")

    if data.isnull().values.any():
        print("数据中存在缺失值")
        return

    with open(args.pkl_data, 'wb') as f:
        pickle.dump(data, f)
    print(f"已保存{args.pkl_data}")

    num_train = int(len(data) * args.train_ratio)
    num_valid = int(len(data) * args.valid_ratio)
    num_test = len(data) - num_train - num_valid

    indices = list(range(len(data)))
    train_indices, valid_indices, test_indices = torch.utils.data.random_split(indices,[num_train, num_valid, num_test])

    train_data = data.iloc[train_indices.indices]
    valid_data = data.iloc[valid_indices.indices]
    test_data = data.iloc[test_indices.indices]

    print(f"训练集大小: {len(train_data)}, 验证集大小: {len(valid_data)}, 测试集大小: {len(test_data)}")

    def print_label_distribution(df, name):
        label_counts = df[label_col].value_counts()
        print(f"\n[{name}集标签分布]")
        print(f"正样本: {label_counts.get(1, 0)} 个 (占比: {label_counts.get(1, 0) / len(df):.2%})")
        print(f"负样本: {label_counts.get(0, 0)} 个 (占比: {label_counts.get(0, 0) / len(df):.2%})")

    print_label_distribution(train_data, "训练")
    print_label_distribution(valid_data, "验证")
    print_label_distribution(test_data, "测试")

    train_dataset = CustomDataset(train_data)
    valid_dataset = CustomDataset(valid_data)
    test_dataset = CustomDataset(test_data)

    torch.save(train_dataset, args.train_dataset)
    torch.save(valid_dataset, args.valid_dataset)
    torch.save(test_dataset, args.test_dataset)

    print("数据集已制作并保存")



# check_dataset
def check_dataset(num_samples=5):
    try:
        dataset = torch.load(args.train_dataset, weights_only=False)
        print(f"数据集 {args.train_dataset} 加载成功")
    except Exception as e:
        print(f"数据集加载失败，错误信息：{str(e)}")
        return

    print("\n=== 基础信息验证 ===")
    print(f"数据集长度：{len(dataset)} 条样本")

    try:
        sample_text, sample_label = dataset[0]
        print(f"样本结构正确 (text, label)")
    except Exception as e:
        print(f"样本结构错误：{str(e)}")
        return

    print("\n=== 文本格式验证 ===")
    if not isinstance(sample_text, str):
        print(f"文本类型错误，应为str，实际得到：{type(sample_text)}")
    else:
        print(f"文本类型正确 (str)")
        print(f"示例文本长度：{len(sample_text)} 字符")
        print(f"示例文本前50字符：{sample_text[:50]}...")

    print("\n=== 标签格式验证 ===")
    if not isinstance(sample_label, torch.Tensor):
        print(f"标签类型错误，应为torch.Tensor，实际得到：{type(sample_label)}")
    else:
        print(f"标签类型正确 (torch.Tensor)")
        if sample_label.dim() != 0:
            print(f"标签维度错误，应为标量，实际维度：{sample_label.shape}")
        else:
            print(f"标签维度正确 (标量)")
        print(f"标签值范围：{torch.min(sample_label)} ~ {torch.max(sample_label)} (应为0.0或1.0)")
        print(f"标签示例值：{sample_label.item()}")  # 使用item()获取标量值

    print("\n=== 随机样本抽查 ===")
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    for i in indices:
        text, label = dataset[i]
        print(f"\n样本 {i}:")
        print(f"文本长度：{len(text)}")
        print(f"标签值：{label.item()}")

    print("\n=== 全量标签统计 ===")
    all_labels = torch.stack([label for _, label in dataset])
    pos_count = all_labels.sum().item()
    neg_count = len(all_labels) - pos_count
    total = len(all_labels)
    print(f"正样本数：{int(pos_count)} (占比: {pos_count / (total + 1e-7):.2%})")
    print(f"负样本数：{neg_count} (占比: {neg_count / (total + 1e-7):.2%})")

    if all_labels.dim() != 1:
        print(f"标签维度错误，应为1维张量，实际维度：{all_labels.dim()}")
    else:
        print("标签维度正确 (1维张量)")

    if not torch.all(all_labels.eq(0) | all_labels.eq(1)):
        print("发现非0/1标签值")
    else:
        print("所有标签值为0或1")



# BertClassifier
class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained('./model/guwenbert-base')
        self.hidden_dim = AutoConfig.from_pretrained('./model/guwenbert-base').hidden_size
        self.tokenizer = AutoTokenizer.from_pretrained('./model/guwenbert-base')
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def forward(self, sent):
        input_ids, attention_mask = self.prepare_inputs(sent)
        encoding = self.bert(input_ids=input_ids,
                             attention_mask=attention_mask).last_hidden_state[:, 0, :]
        logits = self.classifier(encoding).squeeze(-1)
        return logits

    def prepare_inputs(self, sent):
        inputs = self.tokenizer(sent, max_length=256, padding='longest', truncation=True, return_tensors='pt').to(
            self.device)
        return inputs['input_ids'].to(self.device), inputs['attention_mask'].to(self.device)



# calculate_metrics
def calculate_metrics(preds, labels, threshold=0.5):
    probs = torch.sigmoid(torch.tensor(preds)).numpy()
    preds_bin = (probs >= threshold).astype(int)

    cm = confusion_matrix(labels, preds_bin, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        'acc': accuracy_score(labels, preds_bin),
        'f1': f1_score(labels, preds_bin, zero_division=0),
        'precision': precision_score(labels, preds_bin, zero_division=0),
        'recall': recall_score(labels, preds_bin, zero_division=0),
        'auc': roc_auc_score(labels, probs),
        'cm': cm,
        'tn': tn, 'fp': fp,
        'fn': fn, 'tp': tp
    }



# compute_valid_metrics
def compute_valid_metrics(model, data_loader, device, criterion):
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0

    with torch.no_grad():
        for texts, labels in data_loader:
            labels = labels.float().to(device)
            logits = model(texts)

            loss = criterion(logits, labels)
            total_loss += loss.item()

            all_preds.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return {
        'loss': total_loss / len(data_loader),
        **calculate_metrics(np.concatenate(all_preds), np.concatenate(all_labels))
    }



# train
def train():
    print("开始加载数据集并初始化")
    train_dataset = torch.load(args.train_dataset, weights_only=False)
    valid_dataset = torch.load(args.valid_dataset, weights_only=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    plt.ion()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))
    fig.subplots_adjust(wspace=0.2)
    axes = [ax1, ax2, ax3, ax4]
    for ax in axes:
        ax.set_box_aspect(0.8)

    metric_history = {
        'train_loss': [], 'valid_loss': [],
        'train_acc': [], 'valid_acc': [],
        'train_f1': [], 'valid_f1': [],
        'train_auc': [], 'valid_auc': []
    }

    model = BertClassifier()
    model.to(model.device)
    print(model.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    pos_weights = torch.tensor(1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(model.device))

    best_metrics = {
        'loss': float('inf'),
        'f1': -float('inf')
    }

    print("初始化完毕")
    print("开始训练模型")

    for epoch in range(args.num_epochs):

        pbar = tqdm(desc='Training Epoch %d' % (epoch + 1), total=len(train_loader))
        model.train()
        epoch_preds, epoch_labels, total_loss = [], [], 0.0

        for texts, labels in train_loader:
            labels = labels.float().to(model.device)

            optimizer.zero_grad()
            logits = model(texts)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            epoch_preds.append(logits.detach().cpu().numpy())
            epoch_labels.append(labels.cpu().numpy())

            pbar.update(1)
            pbar.set_postfix_str('Loss: %.2f' % loss.item())

        train_metrics = {
            'loss': total_loss / len(train_loader),
            **calculate_metrics(np.concatenate(epoch_preds), np.concatenate(epoch_labels))
        }

        pbar.close()

        valid_metrics = compute_valid_metrics(model, valid_loader, model.device, criterion)

        for key in ['loss', 'acc', 'f1', 'auc']:
            metric_history[f'train_{key}'].append(train_metrics[key])
            metric_history[f'valid_{key}'].append(valid_metrics[key])

        epochs = range(1, epoch + 2)
        titles = ['Loss Curve', 'Accuracy', 'F1 Score', 'AUC']
        y_labels = ['Loss', 'Accuracy', 'F1 Score', 'AUC']
        data_pairs = [
            (metric_history['train_loss'], metric_history['valid_loss']),
            (metric_history['train_acc'], metric_history['valid_acc']),
            (metric_history['train_f1'], metric_history['valid_f1']),
            (metric_history['train_auc'], metric_history['valid_auc'])
        ]
        for ax, title, ylabel, (train_data, valid_data) in zip(axes, titles, y_labels, data_pairs):
            ax.clear()
            ax.plot(epochs, train_data, 'b-', label='Train')
            ax.plot(epochs, valid_data, 'r-', label='Valid')
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.legend()
            if ylabel != 'Loss':
                ax.set_ylim(0, 1)

        plt.pause(0.1)

        torch.save(model.state_dict(), args.save_epoch_checkpoint.format(epoch + 1))

        current_loss = valid_metrics['loss']
        current_f1 = valid_metrics['f1']
        if (current_loss <= best_metrics['loss']) and (current_f1 >= best_metrics['f1']):
            best_metrics['loss'] = current_loss
            best_metrics['f1'] = current_f1
            torch.save(model.state_dict(), args.save_autobest_checkpoint)
            print(f"Epoch {epoch+1}: 发现最佳模型，验证集Loss={current_loss:.4f}, F1={current_f1:.4f}")

        print(f"\n{'=' * 20}")
        print(f"Epoch {epoch+1}/{args.num_epochs} 综合报告")

        print("\n[训练集]")
        print(f"Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['acc']:.2%}")
        print(f"F1: {train_metrics['f1']:.4f} | AUC: {train_metrics['auc']:.4f}")
        print(f"Precision: {train_metrics['precision']:.4f} | Recall: {train_metrics['recall']:.4f}")
        print(f"混淆矩阵指标:")
        print(f"TN={train_metrics['tn']} FP={train_metrics['fp']}")
        print(f"FN={train_metrics['fn']} TP={train_metrics['tp']}")

        print("\n[验证集]")
        print(f"Loss: {valid_metrics['loss']:.4f} | Acc: {valid_metrics['acc']:.2%}")
        print(f"F1: {valid_metrics['f1']:.4f} | AUC: {valid_metrics['auc']:.4f}")
        print(f"\n[验证集当前最佳] Loss: {best_metrics['loss']:.4f} | F1: {best_metrics['f1']:.4f}")
        print(f"Precision: {valid_metrics['precision']:.4f} | Recall: {valid_metrics['recall']:.4f}")
        print(f"混淆矩阵指标:")
        print(f"TN={valid_metrics['tn']} FP={valid_metrics['fp']}")
        print(f"FN={valid_metrics['fn']} TP={valid_metrics['tp']}")
        print('=' * 20 + '\n')

    plt.savefig('training_results.png')
    print("模型训练完成")
    plt.ioff()
    plt.show()



# check_test_dataset
def check_test_dataset():
    test_dataset = torch.load(args.test_dataset, weights_only=False)

    if not isinstance(test_dataset, CustomDataset):
        raise ValueError("test_dataset不是CustomDataset的实例")

    print(f'测试集大小: {len(test_dataset)}')

    for i in range(min(5, len(test_dataset))):
        text, label = test_dataset[i]
        print(f'Sample {i}: Text: {text}, Label: {label.item()}')



# infer_test_dataset
def infer_test_dataset():
    print("\n开始推理测试集...")

    test_dataset = torch.load(args.test_dataset, weights_only=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = BertClassifier()
    model.load_state_dict(torch.load(args.best_model))
    model.to(model.device)
    model.eval()

    input_texts = []
    true_labels = []
    pred_logits = []

    with torch.no_grad():
        for texts, labels in tqdm(test_loader, desc="推理进度"):
            logits = model(texts)

            input_texts.extend(texts)
            true_labels.extend(labels.cpu().numpy())
            pred_logits.extend(logits.cpu().numpy())

    pred_probs = torch.sigmoid(torch.tensor(pred_logits)).numpy()
    pred_labels = (pred_probs >= 0.5).astype(int)

    df = pd.DataFrame({
        'memorial': input_texts,
        'isee': true_labels,
        'predict_isee': pred_labels.flatten()
    })

    df.to_excel(args.infer_file, index=False, engine='openpyxl')
    print(f"推理结果已保存至: {args.infer_file}")

    metrics = calculate_metrics(pred_logits, true_labels)
    print("\n=== 测试集评估报告 ===")
    print(f"Acc: {metrics['acc']:.2%}")
    print(f"F1: {metrics['f1']:.4f} | AUC: {metrics['auc']:.4f}")
    print(f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")
    print(f"混淆矩阵指标:")
    print(f"TN={metrics['tn']} FP={metrics['fp']}")
    print(f"FN={metrics['fn']} TP={metrics['tp']}")



# InferenceDataset
class InferenceDataset(Dataset):
    def __init__(self, data):
        assert 'id_tbgg' in data.columns, "缺失id_tbgg列"
        assert 'memorial' in data.columns, "缺失memorial列"
        self.ids = data['id_tbgg'].tolist()
        self.texts = data['memorial'].astype(str).tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.ids[idx], self.texts[idx]



# infer_full_dataset
def infer_full_dataset():
    """端到端的全样本推理流程"""
    print("\n开始全样本推理流程...")

    def process_inferdata():
        print("数据预处理中...")
        df = pd.read_excel(args.memorial_data, engine='openpyxl')

        full_dataset = InferenceDataset(df)
        torch.save(full_dataset, args.full_dataset)
        print(f"预处理数据已保存至 {args.full_dataset}")

    try:
        if os.path.exists(args.full_dataset):
            print(f"发现已预处理文件 {args.full_dataset}")
        else:
            process_inferdata()

        dataset = torch.load(args.full_dataset, weights_only=False)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=lambda batch: (
                [item[0] for item in batch],
                [item[1] for item in batch]
            )
        )

        print("加载模型中...")
        model = BertClassifier()
        model.load_state_dict(torch.load(args.best_model))
        model.to(model.device)
        model.eval()

        print("开始批量推理...")
        results = []
        with torch.no_grad():
            for ids, texts in tqdm(loader, desc="推理进度", unit="batch"):
                logits = model(texts)
                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs >= 0.5).astype(int)

                batch_results = zip(ids, texts, preds)
                results.extend([{
                    'id_tbgg': i,
                    'memorial': t,
                    'predict_isee': p.item(),
                } for i, t, p in batch_results])

        result_df = pd.DataFrame(results)
        result_df.to_excel(args.output_dataset, index=False, engine='openpyxl')
        print(f"推理完成！结果已保存至 {args.output_dataset}")

    except Exception as e:
        print(f"推理流程异常终止: {str(e)}")
        raise






if __name__ == "__main__":
    if args.process_data:
        process_data()

    if args.check_dataset:
        check_dataset()

    if args.train:
        train()

    if args.check_test_dataset:
        check_test_dataset()

    if args.infer_test_dataset:
        infer_test_dataset()

    if args.infer_full_dataset:
        infer_full_dataset()

