import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from rdkit import Chem
import torch.optim as optim
from model_schnet import GNNModel
# from dataset_v2 import MoleculeACDataset_v2
# from dataset_v2_test import MoleculeACDataset_v2
from dataset_v2_test_schnet import MoleculeACDataset_v2
import torch.nn as nn
import random, os
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from tqdm import tqdm
from datetime import datetime
import logging
from tools import calculate_metrics
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import  random_split
import math

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse

import pdb

# 设置命令行参数
parser = argparse.ArgumentParser(description="Train a model with AdamW optimizer and learning rate scheduler.")

# 添加label_name的命令行参数
parser.add_argument(
    '--label_name', 
    type=str, 
    default='None', 
    choices=['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv'], 
    required=True,
    help='The label name to be used in training. Must be one of: mu, alpha, homo, lumo, gap, r2, zpve, U0, U, H, G, Cv.'
)

# 解析命令行参数
args = parser.parse_args()

# v2版本加入进化path

# Set up logging
# List of Python files to log
files_to_log = [
    '/home/data1/lk/project/mol_tree/train_v2_qm9_schnet.py',
    '/home/data1/lk/project/mol_tree/tools.py',
    '/home/data1/lk/project/mol_tree/model_schnet.py',
    # '/home/data1/lk/project/mol_tree/dataset_v2.py'
    "/home/data1/lk/project/mol_tree/dataset_v2_test_schnet.py"
]

def extract_model_name(files_to_log):
    for file_path in files_to_log:
        if 'train_v2_qm9_' in file_path:
            return file_path.split('train_v2_qm9_')[-1].split('.py')[0]
    return 'unknown_model'

# 提取模型名称
model_name = extract_model_name(files_to_log)

# label_name in mu,alpha,homo,lumo,gap,r2,zpve,U0,U,H,G,Cv
label_name = args.label_name

log_dir = os.path.join('logs', model_name, label_name)
os.makedirs(log_dir, exist_ok=True)
import pytz
china_tz = pytz.timezone('Asia/Shanghai')
start_time = datetime.now(china_tz).strftime("%Y%m%d_%H%M%S")
now_time = start_time
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# 如果需要resume 需要手动设置一下start_time
start_time = '20250111_225140'  # -- r2
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
log_file = os.path.join(log_dir, f'schnet_qm9_{label_name}_training_log_{start_time}.log')
logging.basicConfig(filename=log_file, level=logging.INFO)

# Function to log messages
def log(message, noprint=False):
    
    if noprint==False:
        print(message)  # Print to console
        
    logging.info(message)  # Save to log file

# Function to log the content of specified Python files
def log_python_file(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            log(f"Logging content of the file: {file_path}")
            log(content, noprint=True)
    except Exception as e:
        log(f"Error while reading the file {file_path}: {e}")

if now_time == start_time:
    # Log the content of each file
    for file in files_to_log:
        log_python_file(file)

# Set a random seed
seed = 42
log(f'seed is {seed}')

# Python random seed
random.seed(seed)
# NumPy random seed
np.random.seed(seed)
# PyTorch random seed
torch.manual_seed(seed)

# For GPU (if using CUDA)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For all GPUs

# If you are using cuDNN (for GPU training)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


import json
import random

# File paths
input_file = "/home/data1/lk/project/mol_tree/graph/evolution_graph_133885_['edit_distance', 'graph']_0.3_v2.json"
output_file = "/home/data1/lk/project/mol_tree/graph/test_smiles.txt"
dataset_file = '/home/data1/lk/project/mol_generate/GDSS/data/qm9.csv'

# Load JSON file
with open(input_file, 'r') as f:
    data = json.load(f)

# Extract SMILES from the 'nodes' list
smiles_list = data.get('nodes', [])

# Select 5% of SMILES randomly
selected_smiles = random.sample(smiles_list, max(1, int(len(smiles_list) * 0.1)))

# Save to the output file, one SMILES per line
with open(output_file, 'w') as f:
    for smile in selected_smiles:
        f.write(smile + '\n')




# 加载数据集
dataset = MoleculeACDataset_v2(input_file, dataset_file, test_file = output_file, mode='trainval', label_name = label_name)

label_logarithmic_transformation = dataset.label_logarithmic_transformation
log(f"Label logarithmic transformation is:{label_logarithmic_transformation}.")


# 计算训练、验证和测试集的大小
train_size = int( 8/9 * len(dataset))
val_size = len(dataset) - train_size  # 15% for validation

# 随机分割数据集
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


test_dataset = MoleculeACDataset_v2(input_file, dataset_file, test_file = output_file, mode='test', label_name = label_name)
test_dataloader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=0)

# 创建数据加载器
train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=0)


# 初始化模型和优化器
# modify
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # cuda:0 cuda:1 cuda:2
is_mu = label_name =='mu'
model = GNNModel(device, is_mu=is_mu).to(device)  # Move model to GPU
# optimizer = optim.Adam(model.parameters(), lr=0.001)
if label_name != 'Cv':
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
else:
    optimizer = optim.AdamW(model.parameters(), lr=0.00085, weight_decay=1e-4)

# 学习率调度器
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1, min_lr=1e-9)


# 0.002 for homo 8192
# 

# criterion = nn.MSELoss()
criterion = nn.L1Loss()
# scheduler = StepLR(optimizer, step_size=50, gamma=0.9)  # 每 10 个 epoch 将学习率缩小为原来的 0.1 倍


def save_checkpoint(epoch, model, optimizer, scheduler, best_val_loss, checkpoint_dir, start_time):
    # 将模型移动到 CPU
    model.to('cpu')
    
    # 保存检查点
    pth_file = os.path.join(checkpoint_dir, f'gnn_model_epoch_{start_time}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
    }, pth_file)
    
    # 将模型移回 GPU
    model.to(device)
    
    log(f"Checkpoint saved at {pth_file}")
    
def load_checkpoint(model, optimizer, scheduler, checkpoint_dir, start_time):
    pth_file = os.path.join(checkpoint_dir, f'gnn_model_epoch_{start_time}.pth')
    if os.path.exists(pth_file):
        # 加载检查点
        checkpoint = torch.load(pth_file, map_location='cpu')  # 先加载到 CPU
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        
        # 将模型和优化器状态移动到 GPU
        model.to(device)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        
        log(f"Loaded checkpoint from {pth_file}, epoch {checkpoint['epoch']+1}, best_val_loss {best_val_loss:.6f}")
        return start_epoch, best_val_loss
    else:
        log("No checkpoint found, starting from scratch.")
        return 0, float('inf')

# 设置检查点目录
checkpoint_dir = os.path.join('checkpoints', model_name, label_name)
os.makedirs(checkpoint_dir, exist_ok=True)
log(f"Checkpoint directory: {checkpoint_dir}")
            
def predict(model, dataloader, mode=None):
    model.eval()
    total_loss = 0
    total_loss1 = 0
    total_loss2 = 0
    all_targets = []
    all_outputs = []
    
    pdb.set_trace()

    with torch.no_grad():
        for graph1, graph2, target, paths, paths_labels, paths_labels_masks in dataloader:
            graph1 = graph1.to(device)  # Move data to GPU
            graph2 = graph2.to(device)  # Move data to GPU
            target = target.to(device)  # Move target to GPU
            output = model.forward_with_path(graph1, graph2, paths, paths_labels, paths_labels_masks)

            # 分别计算两个回归项的损失
            loss1 = criterion(output[:, 0].float(), target[:, 0].float())
            loss2 = criterion(output[:, 1].float(), target[:, 1].float())
            
            # modify
            # homo gap zpve mu alpha r2
            loss = loss1 * 0.9  + loss2 * 0.1
            # lumo Cv
            # loss = loss1 + loss2

            total_loss += loss.item()
            total_loss1 += loss1.item()
            total_loss2 += loss2.item()

            # if mode == 'test':
            #     all_targets.append(target.cpu().numpy())  # Store on CPU
            #     all_outputs.append(output.cpu().numpy())  # Store both outputs

    avg_loss = total_loss / len(dataloader)
    avg_loss1 = total_loss1 / len(dataloader)
    avg_loss2 = total_loss2 / len(dataloader)

    # # 计算相关系数和R²
    # if mode == 'test':
    #     all_targets = np.concatenate(all_targets)  
    #     all_outputs = np.concatenate(all_outputs)
    #     # 分别提取每个回归项的目标和输出
    #     target1 = all_targets[:, 0]
    #     target2 = all_targets[:, 1]
    #     output1 = all_outputs[:, 0]
    #     output2 = all_outputs[:, 1]

    #     # 计算第一个输出的 R² 和 Pearson
    #     pearson_corr1 = pearsonr(target1, output1)[0]
    #     r2_1 = r2_score(target1, output1)

    #     # 计算第二个输出的 R² 和 Pearson
    #     pearson_corr2 = pearsonr(target2, output2)[0]
    #     r2_2 = r2_score(target2, output2)

    #     log(f'Output 1 - R² Score: {r2_1:.6f}, Pearson Correlation: {pearson_corr1:.6f}')
    #     log(f'Output 2 - R² Score: {r2_2:.6f}, Pearson Correlation: {pearson_corr2:.6f}')

    return avg_loss, avg_loss1, avg_loss2

def infer(model, dataloader, mode=None):
    model.eval()
    total_loss = 0
    total_loss1 = 0
    total_loss2 = 0
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for iter, (graph1, graph2, target, paths, paths_labels, paths_labels_masks) in enumerate(dataloader):
            graph1 = graph1.to(device)  # Move data to GPU
            graph2 = graph2.to(device)  # Move data to GPU
            target = target.to(device)  # Move target to GPU
            output = model.forward_with_path(graph1, graph2, paths, paths_labels, paths_labels_masks)
            
            # 分别计算两个回归项的损失
            loss1 = criterion(output[:, 0].float(), target[:, 0].float())
            loss2 = criterion(output[:, 1].float(), target[:, 1].float())
            
            # modify
            # homo gap zpve mu alpha r2
            loss = loss1 * 0.9  + loss2 * 0.1
            # lumo Cv
            # loss = loss1 + loss2

            total_loss += loss.item()
            total_loss1 += loss1.item()
            total_loss2 += loss2.item()
                
            if mode == 'test':
                all_targets.append(target.cpu().numpy())  # Store on CPU
                all_outputs.append(output.cpu().numpy())  # Store both outputs

                
    avg_loss = total_loss / len(dataloader)
    avg_loss1 = total_loss1 / len(dataloader)
    avg_loss2 = total_loss2 / len(dataloader)

    # 计算相关系数和R²
    if mode == 'test':
        all_targets = np.concatenate(all_targets)  
        all_outputs = np.concatenate(all_outputs)
        # 分别提取每个回归项的目标和输出
        target1 = all_targets[:, 0]
        target2 = all_targets[:, 1]
        target3 = all_targets[:, 2]
        
        output1 = all_outputs[:, 0]
        output2 = all_outputs[:, 1]

    # metrics1 = calculate_metrics(dataset.denormalize(target1), dataset.denormalize(output1))
    # metrics2 = calculate_metrics(dataset.denormalize(target2), dataset.denormalize(output2))
    # # if label_logarithmic_transformation:
    # #     # import pdb;pdb.set_trace()
    # #     temp_target2_pred = np.exp(output1+np.log(target3+0.00001))-0.00001
    # #     metrics3 = calculate_metrics(target2, temp_target2_pred)
    # # else:
    # # import pdb;pdb.set_trace()
    # metrics3 = calculate_metrics(dataset.denormalize(target2), dataset.denormalize( ( output1 *  (dataset.label_max_ratio - dataset.label_min_ratio) + dataset.label_min_ratio + 1 ) * target3 ))

    metrics1 = calculate_metrics(target1, output1)
    metrics2 = calculate_metrics(target2, output2)
    # modify
    # zpve
    # metrics3 = calculate_metrics(target2, ( output1 + 1 ) * target3)
    # homo lumo gap mu alpha r2 Cv
    metrics3 = calculate_metrics(target2, output1 + target3)
    # 打印结果
    log(f"Output 1 - R² Score: {metrics1['R2']:.6f}, Pearson Correlation: {metrics1['Pearson Correlation']:.6f}, MSE: {metrics1['MSE']:.6f}, MAE: {metrics1['MAE']:.6f}, Rank Loss: {metrics1['Rank Loss']:.6f}")
    log(f"Output 2 - R² Score: {metrics2['R2']:.6f}, Pearson Correlation: {metrics2['Pearson Correlation']:.6f}, MSE: {metrics2['MSE']:.6f}, MAE: {metrics2['MAE']:.6f}, Rank Loss: {metrics2['Rank Loss']:.6f}")
    log(f"Output 3 - R² Score: {metrics3['R2']:.6f}, Pearson Correlation: {metrics3['Pearson Correlation']:.6f}, MSE: {metrics3['MSE']:.6f}, MAE: {metrics3['MAE']:.6f}, Rank Loss: {metrics3['Rank Loss']:.6f}")

    return avg_loss, avg_loss1, avg_loss2

def train(model, train_dataloader, val_dataloader, optimizer, scheduler, criterion, epochs=1000, resume=False):
    # 如果 resume 为 True，尝试加载检查点
    if resume:
        start_epoch, best_val_loss = load_checkpoint(model, optimizer, scheduler, checkpoint_dir, start_time)
    else:
        start_epoch, best_val_loss = 0, float('inf')
    
    for epoch in tqdm(range(start_epoch, epochs),desc=f'for {label_name}'):
        model.train()
        total_loss = 0
        total_loss1 = 0
        total_loss2 = 0
        
        for iter, (graph1, graph2, target, paths, paths_labels, paths_labels_masks) in enumerate(train_dataloader):
            optimizer.zero_grad()
            graph1 = graph1.to(device)  # Move data to GPU
            graph2 = graph2.to(device)  # Move data to GPU
            target = target.to(device)  # Move target to GPU
            
            output = model.forward_with_path(graph1, graph2, paths, paths_labels, paths_labels_masks)
            
            loss1 = criterion(output[:, 0].squeeze(), target[:, 0].float())
            loss2 = criterion(output[:, 1].squeeze(), target[:, 1].float())
            
            # modify
            # homo gap zpve mu alpha r2
            loss = loss1 * 0.9  + loss2 * 0.1
            # lumo Cv
            # loss = loss1 + loss2
            
            loss.backward()
            optimizer.step()
            
            
            total_loss += loss.item()
            total_loss1 += loss1.item()
            total_loss2 += loss2.item()
            
            log(f'Epoch [{epoch + 1}/{epochs}], {label_name} Iter [{iter + 1}/{len(train_dataloader)}], Loss: {loss.item():.6f}, Loss1: {loss1.item():.6f}, Loss2: {loss2.item():.6f}')

            
        avg_loss = total_loss / len(train_dataloader)
        avg_loss1 = total_loss1 / len(train_dataloader)
        avg_loss2 = total_loss2 / len(train_dataloader)

        log(f'Epoch [{epoch + 1}/{epochs}], {label_name} Loss: {avg_loss:.6f}, Loss1: {avg_loss1:.6f}, Loss2: {avg_loss2:.6f}')
        
        # Validate
        val_loss, val_loss1, val_loss2 = predict(model, val_dataloader)
        log(f'Epoch [{epoch + 1}/{epochs}], {label_name} Val Loss: {val_loss:.6f}, Val Loss1: {val_loss1:.6f}, Val Loss2: {val_loss2:.6f}')
        
        # 更新学习率
        scheduler.step(val_loss)
        # Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        log(f'Epoch [{epoch + 1}/{epochs}], Current LR: {current_lr:.8e}')
        
        # Check if validation loss improved and save model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            test_loss, test_loss1, test_loss2 = infer(model, test_dataloader, mode='test')
            # test_loss, test_loss1, test_loss2 = predict(model, test_dataloader, mode='test')
            # log(f'Test Loss: {test_loss:.6f}, Test Loss1: {test_loss1:.6f}, Test Loss2: {test_loss2:.6f}')
            
            save_checkpoint(epoch, model, optimizer, scheduler, best_val_loss, checkpoint_dir, start_time)
            

# 训练模型
# train(model, train_dataloader, val_dataloader, optimizer, scheduler, criterion, epochs=800, resume=False)
train(model, train_dataloader, val_dataloader, optimizer, scheduler, criterion, epochs=800, resume=True)

# 更改学习率之后 要更改对应的train predict infer的metrics3和loss  和  dataset_v2_test_schnet.py中的target  crtl + f # modify