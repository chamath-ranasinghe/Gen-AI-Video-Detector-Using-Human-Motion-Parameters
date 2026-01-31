#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI-Generated Video Detector
Trains a deep learning model to classify videos as real or AI-generated
based on SMPL-X body model parameters and visual features
"""

import os
import os.path as osp
import numpy as np
import argparse
from tqdm import tqdm
from typing import List, Tuple, Dict
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt


class VideoParameterDataset(Dataset):
    """Dataset for video SMPL-X parameters"""
    
    def __init__(self, param_files: List[str], labels: List[int], 
                 max_frames: int = 100, augment: bool = False):
        """
        Args:
            param_files: List of paths to .npz files containing frame parameters
            labels: List of labels (0=real, 1=AI-generated)
            max_frames: Maximum number of frames to use per video
            augment: Whether to apply data augmentation
        """
        self.param_files = param_files
        self.labels = labels
        self.max_frames = max_frames
        self.augment = augment
        
    def __len__(self):
        return len(self.param_files)
    
    def __getitem__(self, idx):
        # Load parameters
        data = np.load(self.param_files[idx], allow_pickle=True)
        params = data['params']
        
        # Extract features from each frame
        frame_features = []
        for frame_param in params[:self.max_frames]:
            features = self._extract_features(frame_param)
            frame_features.append(features)
        
        # Pad or truncate to max_frames
        if len(frame_features) < self.max_frames:
            # Pad with zeros
            padding = [np.zeros_like(frame_features[0])] * (self.max_frames - len(frame_features))
            frame_features.extend(padding)
        
        frame_features = np.stack(frame_features[:self.max_frames])
        
        # Data augmentation (temporal jittering)
        if self.augment and np.random.random() > 0.5:
            frame_features = self._temporal_jitter(frame_features)
        
        return {
            'features': torch.FloatTensor(frame_features),
            'label': torch.LongTensor([self.labels[idx]])[0],
            'num_frames': min(len(params), self.max_frames)
        }
    
    def _extract_features(self, frame_param: dict) -> np.ndarray:
        """Extract relevant features from frame parameters"""
        features = []
        
        # Helper function to safely add features
        def add_feature(value, max_len=None):
            if np.isscalar(value):
                features.append(float(value))
            else:
                value_flat = np.array(value).flatten()
                if max_len is not None:
                    value_flat = value_flat[:max_len]
                features.extend([float(v) for v in value_flat])
        
        # Camera parameters
        if 'camera_scale' in frame_param:
            add_feature(frame_param['camera_scale'])
        if 'camera_translation' in frame_param:
            add_feature(frame_param['camera_translation'])
        
        # Body pose parameters (typically 63 or 69 dimensions)
        if 'body_pose' in frame_param:
            add_feature(frame_param['body_pose'])
        
        # Global orientation (3 dimensions)
        if 'global_orient' in frame_param:
            add_feature(frame_param['global_orient'])
        
        # Shape parameters (typically 10 dimensions)
        if 'betas' in frame_param:
            add_feature(frame_param['betas'])
        
        # Hand pose parameters (limit to 15 each to avoid huge feature vectors)
        if 'left_hand_pose' in frame_param:
            add_feature(frame_param['left_hand_pose'], max_len=15)
        if 'right_hand_pose' in frame_param:
            add_feature(frame_param['right_hand_pose'], max_len=15)
        
        # Jaw pose and expression
        if 'jaw_pose' in frame_param:
            add_feature(frame_param['jaw_pose'])
        if 'expression' in frame_param:
            add_feature(frame_param['expression'], max_len=10)
        
        return np.array(features, dtype=np.float32)
    
    def _temporal_jitter(self, features: np.ndarray) -> np.ndarray:
        """Apply temporal jittering for augmentation"""
        # Randomly shift temporal sequence
        shift = np.random.randint(-5, 6)
        if shift > 0:
            features = np.concatenate([features[shift:], np.zeros((shift, features.shape[1]))])
        elif shift < 0:
            features = np.concatenate([np.zeros((-shift, features.shape[1])), features[:shift]])
        return features


class DeepfakeDetectorLSTM(nn.Module):
    """LSTM-based detector for temporal sequence analysis"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, 
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)  # Binary classification
        )
    
    def forward(self, x, lengths=None):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        
        # Apply attention mechanism
        attn_weights = self.attention(lstm_out)
        attended = torch.sum(lstm_out * attn_weights, dim=1)
        
        # Classify
        logits = self.classifier(attended)
        return logits


class DeepfakeDetectorTransformer(nn.Module):
    """Transformer-based detector"""
    
    def __init__(self, input_dim: int, d_model: int = 256, 
                 nhead: int = 8, num_layers: int = 4, dropout: float = 0.3):
        super().__init__()
        
        self.embedding = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )
    
    def forward(self, x, lengths=None):
        # x: (batch, seq_len, features)
        # Transpose to (seq_len, batch, features) for PyTorch Transformer
        x = self.embedding(x)
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        x = self.transformer(x)
        x = x.transpose(0, 1)  # Back to (batch, seq_len, d_model)
        # Global average pooling
        x = torch.mean(x, dim=1)
        logits = self.classifier(x)
        return logits


def train_epoch(model: nn.Module, dataloader: DataLoader, 
                criterion, optimizer, device: torch.device) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        features = batch['features'].to(device)
        labels = batch['label'].to(device)
        lengths = batch['num_frames']
        
        optimizer.zero_grad()
        outputs = model(features, lengths)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model: nn.Module, dataloader: DataLoader, 
            criterion, device: torch.device) -> Tuple[float, Dict]:
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            lengths = batch['num_frames']
            
            outputs = model(features, lengths)
            loss = criterion(outputs, labels)
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )
    auc = roc_auc_score(all_labels, all_probs)
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }
    
    return avg_loss, metrics


def plot_training_history(history: Dict, save_path: str):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(history['val_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # F1 Score
    axes[1, 0].plot(history['val_f1'], label='Validation F1')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Validation F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # AUC
    axes[1, 1].plot(history['val_auc'], label='Validation AUC')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('AUC')
    axes[1, 1].set_title('Validation AUC')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Train deepfake detector using SMPL-X parameters'
    )
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing processed video parameters')
    parser.add_argument('--real-subdir', type=str, default='real',
                       help='Subdirectory name for real videos')
    parser.add_argument('--fake-subdir', type=str, default='fake',
                       help='Subdirectory name for AI-generated videos')
    parser.add_argument('--model-type', type=str, default='lstm',
                       choices=['lstm', 'transformer'],
                       help='Model architecture to use')
    parser.add_argument('--max-frames', type=int, default=100,
                       help='Maximum frames per video')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='Hidden dimension size')
    parser.add_argument('--num-layers', type=int, default=2,
                       help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--output-dir', type=str, default='detector_output',
                       help='Output directory for model and results')
    parser.add_argument('--test-split', type=float, default=0.2,
                       help='Test set split ratio')
    parser.add_argument('--val-split', type=float, default=0.1,
                       help='Validation set split ratio')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    real_dir = osp.join(args.data_dir, args.real_subdir)
    fake_dir = osp.join(args.data_dir, args.fake_subdir)
    
    real_files = [osp.join(real_dir, f, 'all_frame_params.npz') 
                  for f in os.listdir(real_dir) 
                  if osp.isdir(osp.join(real_dir, f))]
    fake_files = [osp.join(fake_dir, f, 'all_frame_params.npz') 
                  for f in os.listdir(fake_dir) 
                  if osp.isdir(osp.join(fake_dir, f))]
    
    all_files = real_files + fake_files
    all_labels = [0] * len(real_files) + [1] * len(fake_files)
    
    print(f"Found {len(real_files)} real videos and {len(fake_files)} fake videos")
    
    # Split data
    train_val_files, test_files, train_val_labels, test_labels = train_test_split(
        all_files, all_labels, test_size=args.test_split, random_state=42, stratify=all_labels
    )
    
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_val_files, train_val_labels, 
        test_size=args.val_split / (1 - args.test_split),
        random_state=42, stratify=train_val_labels
    )
    
    # Create datasets
    train_dataset = VideoParameterDataset(train_files, train_labels, 
                                         args.max_frames, augment=True)
    val_dataset = VideoParameterDataset(val_files, val_labels, args.max_frames)
    test_dataset = VideoParameterDataset(test_files, test_labels, args.max_frames)
    
    # Determine input dimension from first sample
    sample = train_dataset[0]
    input_dim = sample['features'].shape[1]
    print(f"Input feature dimension: {input_dim}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=4)
    
    # Create model
    print(f"Creating {args.model_type} model...")
    if args.model_type == 'lstm':
        model = DeepfakeDetectorLSTM(
            input_dim, args.hidden_dim, args.num_layers, args.dropout
        )
    else:
        model = DeepfakeDetectorTransformer(
            input_dim, args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout
        )
    
    model = model.to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    print("Starting training...")
    history = {
        'train_loss': [], 'val_loss': [],
        'val_accuracy': [], 'val_precision': [],
        'val_recall': [], 'val_f1': [], 'val_auc': []
    }
    
    best_val_f1 = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_auc'].append(val_metrics['auc'])
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
        
        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'model_config': {
                    'input_dim': input_dim,
                    'hidden_dim': args.hidden_dim,
                    'num_layers': args.num_layers,
                    'dropout': args.dropout,
                    'model_type': args.model_type
                }
            }, osp.join(args.output_dir, 'best_model.pth'))
            print(f"Saved best model (F1: {best_val_f1:.4f})")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    checkpoint = torch.load(osp.join(args.output_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Results:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
    print(f"AUC: {test_metrics['auc']:.4f}")
    
    # Save results
    results = {
        'test_metrics': test_metrics,
        'history': history,
        'args': vars(args)
    }
    
    with open(osp.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot training history
    plot_training_history(history, osp.join(args.output_dir, 'training_history.png'))
    
    print(f"\nTraining complete! Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()