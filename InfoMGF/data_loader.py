import os
import torch
from torch_geometric.data import Data, InMemoryDataset
import pickle
import numpy as np
from torch_geometric.utils import dense_to_sparse, dropout_adj
from torch_geometric.transforms import BaseTransform, Compose

class DropEdge(BaseTransform):
    def __init__(self, p=0.2, force_undirected=False):
        super().__init__()
        self.p = p
        self.force_undirected = force_undirected

    def forward(self, data):
        if self.p <= 0:
            return data

        if hasattr(data, 'edge_index_intra'):
            edge_index = data.edge_index_intra
            target_attr = 'edge_index_intra'
        else:
            edge_index = data.edge_index
            target_attr = 'edge_index'
            
        # Apply dropout
        # dropout_adj returns (edge_index, edge_attr)
        edge_index, _ = dropout_adj(
            edge_index, 
            p=self.p, 
            force_undirected=self.force_undirected,
            training=True
        )
        
        setattr(data, target_attr, edge_index)
            
        return data

class FeatureNoise(BaseTransform):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std

    def forward(self, data):
        if self.std <= 0:
            return data
        
        noise = torch.randn_like(data.x) * self.std
        data.x = data.x + noise
        return data

'''
train_dataset = BrainGraphDataset(root='path/to/your/data', split='train')
val_dataset = BrainGraphDataset(root='path/to/your/data', split='val')
test_dataset = BrainGraphDataset(root='path/to/your/data', split='test')

print(f"Train dataset size: {len(train_dataset)}")
print(f"Val dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# 创建DataLoader
from torch_geometric.loader import DataLoader

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
'''
class BrainGraphDataset(InMemoryDataset):
    def __init__(self, root, split='train', label_mode='binary', smart_resample=False, 
                 balance_strategy='downsample_class1', use_original_only=False, 
                 transform=None, pre_transform=None,
                 augment=True, drop_edge_p=0.05, noise_std=0.05):
        self.split = split
        self.label_mode = label_mode  
        self.smart_resample = smart_resample
        self.balance_strategy = balance_strategy
        self.use_original_only = use_original_only
        
        # Setup dynamic augmentation
        transforms_list = []
        if transform is not None:
            transforms_list.append(transform)
            
        if augment and split == 'train':
            print(f"Enabling dynamic augmentation: DropEdge(p={drop_edge_p}), FeatureNoise(std={noise_std})")
            transforms_list.append(DropEdge(p=drop_edge_p))
            transforms_list.append(FeatureNoise(std=noise_std))
            
        if len(transforms_list) > 0:
            transform = Compose(transforms_list)

        # Support K-Fold data files (e.g., train_fold0, val_fold0)
        # 如果 split 包含 'fold', 直接使用; 否则使用旧格式
        if 'fold' in split:
            self.raw_file_path = os.path.join(root, f'processed_{split}.pkl')
        else:
            self.raw_file_path = os.path.join(root, f'processed_{split}.pkl')
        
        cache_suffix = f'{label_mode}'
        if smart_resample:
            cache_suffix += '_smartresample'
        if use_original_only:
            cache_suffix += '_original'
        self.processed_file_path = os.path.join(root, f'processed_brain_graph_{split}_{cache_suffix}.pt')
        
        super().__init__(root, transform, pre_transform)

        if os.path.exists(self.processed_file_path):
            self.data, self.slices = torch.load(self.processed_file_path)
        else:
            self.process()
            self.data, self.slices = torch.load(self.processed_file_path)
    

    @property
    def raw_file_names(self):
        return []  
    
    @property
    def processed_file_names(self):
        return []  
    
    @property
    def raw_paths(self):
        return [self.raw_file_path]
    
    @property
    def processed_paths(self):
        return [self.processed_file_path]
    
    def process(self):
        """处理数据并保存到自定义路径"""
        print(f"Loading data from: {self.raw_file_path}")
        
        with open(self.raw_file_path, 'rb') as f:
            data_dict = pickle.load(f)

        # 首先构建原始数据列表，保留原始标签信息
        data_list_with_original_labels = []
        for sid, graph_data in data_dict.items():
            node_features = torch.from_numpy(graph_data['node_feats']).float()
            raw_label = int(graph_data['label'])
            
            A_intra = graph_data['A_intra']
            A_global = graph_data['A_global']
            edge_index_intra = dense_to_sparse(torch.from_numpy(A_intra).float())[0]
            edge_index_global = dense_to_sparse(torch.from_numpy(A_global).float())[0]

            data = Data(
                x=node_features,
                y=torch.tensor(raw_label, dtype=torch.long),  # 先保留原始标签
                edge_index_intra=edge_index_intra,
                edge_index_global=edge_index_global,
                sid=sid,
                original_label=raw_label  # 额外保存原始标签
            )
            data_list_with_original_labels.append(data)
        
        # 如果需要，过滤掉增强样本（只使用原始样本）
        if self.use_original_only and self.split == 'train':
            original_count = len(data_list_with_original_labels)
            data_list_with_original_labels = [
                data for data in data_list_with_original_labels 
                if '_aug' not in str(data.sid)
            ]
            filtered_count = len(data_list_with_original_labels)
            print(f"Filtered augmented samples: {original_count} → {filtered_count} (removed {original_count - filtered_count} samples)")
        
   
   
        data_list = []
        for data in data_list_with_original_labels:
            if self.label_mode == 'binary':
                mapped = 0 if data.original_label == 0 else 1
                data.y = torch.tensor(mapped, dtype=torch.long)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_file_path)
        print(f"Saved processed data to: {self.processed_file_path}")
    

   