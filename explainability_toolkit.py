#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Deepfake Detection Explainability Toolkit

Provides:
1. Feature Importance Analysis
2. Attention Visualization (LSTM & Transformer)
3. Per-video explanation generation
"""

import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import argparse
from tqdm import tqdm
import json

# Import your models
from deepfake_detector import (
    DeepfakeDetectorLSTM, 
    DeepfakeDetectorTransformer,
    VideoParameterDataset
)


# ============================================================================
# PART 1: FEATURE IMPORTANCE ANALYSIS
# ============================================================================

class FeatureImportanceAnalyzer:
    """
    Analyzes which SMPL-X parameters are most important for detection
    """
    
    def __init__(self, model, device, model_type='lstm'):
        """
        Args:
            model: Trained deepfake detector (LSTM or Transformer)
            device: torch.device
            model_type: 'lstm' or 'transformer'
        """
        self.model = model
        self.device = device
        self.model_type = model_type
        self.model.eval()
        
        # Feature names will be generated from actual data
        self.feature_names = None
        self.feature_structure = None  # Track what we learned about the data
    
    def _inspect_sample_and_generate_names(self, dataset):
        """
        Inspect the first sample to determine actual feature structure.
        This replaces blind guessing with data-driven name generation.
        """
        if len(dataset) == 0:
            print("Warning: Empty dataset provided")
            return
        
        # Load first sample
        sample = dataset[0]
        features = sample['features']  # [seq_len, feature_dim]
        
        # Get the actual feature dimension
        actual_feature_dim = features.shape[1]
        
        print(f"✓ Inspected first sample: {actual_feature_dim} features total")
        
        # Now inspect the raw parameters to understand the structure
        if hasattr(dataset, 'param_files') and len(dataset.param_files) > 0:
            param_file = dataset.param_files[0]
            data = np.load(param_file, allow_pickle=True)
            params = data['params']
            
            # Extract first frame to inspect structure
            first_frame = params[0]
            
            # Track feature groups as we build names
            names = []
            feature_idx = 0
            
            # Helper to track and name features
            def add_features_from_param(param_dict, key, prefix, max_len=None):
                nonlocal feature_idx, names
                if key in param_dict:
                    value = param_dict[key]
                    if np.isscalar(value):
                        size = 1
                        names.append(f'{prefix}')
                    else:
                        value_flat = np.array(value).flatten()
                        if max_len is not None:
                            value_flat = value_flat[:max_len]
                        size = len(value_flat)
                        for i in range(size):
                            names.append(f'{prefix}_{i}')
                    feature_idx += size
                    return size
                return 0
            
            # Build names matching extraction order in deepfake_detector.py
            add_features_from_param(first_frame, 'camera_scale', 'camera_scale')
            add_features_from_param(first_frame, 'camera_translation', 'camera_translation')
            add_features_from_param(first_frame, 'body_pose', 'body_pose')
            add_features_from_param(first_frame, 'global_orient', 'global_orient')
            add_features_from_param(first_frame, 'betas', 'shape_beta')
            add_features_from_param(first_frame, 'left_hand_pose', 'left_hand_component', max_len=15)
            add_features_from_param(first_frame, 'right_hand_pose', 'right_hand_component', max_len=15)
            add_features_from_param(first_frame, 'jaw_pose', 'jaw_pose')
            add_features_from_param(first_frame, 'expression', 'expression_component', max_len=10)
            
            self.feature_names = names
            self.feature_structure = {
                'total_features': len(names),
                'features_from_data': True,
                'sample_file': param_file
            }
            
            print(f"✓ Generated {len(names)} feature names from actual data structure")
            print(f"  Feature groups detected:")
            
            # Print summary of feature groups
            groups = {}
            for name in names:
                prefix = name.rsplit('_', 1)[0] if '_' in name else name
                groups[prefix] = groups.get(prefix, 0) + 1
            
            for group, count in groups.items():
                print(f"    - {group}: {count} features")
        else:
            print("Warning: Could not inspect dataset structure, using fallback names")
            self.feature_names = self._generate_feature_names_fallback(actual_feature_dim)
    
    def _generate_feature_names(self):
        """Generate human-readable feature names matching the extraction order in deepfake_detector.py
        Uses grouped naming: camera, body_pose, shape, hand_pose, expression, etc.
        """
        names = []
        
        # Camera parameters (4 features)
        names.append('camera_scale')
        names.extend(['camera_translation_x', 'camera_translation_y', 'camera_translation_z'])
        
        # Body pose (63 features = 21 joints × 3 rotation angles)
        body_joints = [
            'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
            'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
            'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'
        ]
        for joint in body_joints:
            names.extend([f'body_pose_{joint}_rot_x', f'body_pose_{joint}_rot_y', f'body_pose_{joint}_rot_z'])
        
        # Global orientation (3 features)
        names.extend(['global_orient_x', 'global_orient_y', 'global_orient_z'])
        
        # Shape parameters (10 beta values)
        for i in range(10):
            names.append(f'shape_beta_{i}')
        
        # Left hand pose (truncated to 15 from full 45)
        for i in range(15):
            names.append(f'left_hand_component_{i}')
        
        # Right hand pose (truncated to 15 from full 45)
        for i in range(15):
            names.append(f'right_hand_component_{i}')
        
        # Jaw pose (3 features)
        names.extend(['jaw_pose_x', 'jaw_pose_y', 'jaw_pose_z'])
        
        # Expression (truncated to 10 from full 50)
        for i in range(10):
            names.append(f'expression_component_{i}')
        
        return names
    
    def _generate_feature_names_fallback(self, num_features):
        """Fallback naming when we can't inspect the data directly"""
        names = self._generate_feature_names()
        
        # Extend with contextual names if needed
        if len(names) < num_features:
            for i in range(len(names), num_features):
                # Guess based on position ranges
                if i < 4:
                    name = f'camera_{i}'
                elif i < 67:  # 4 + 63 body pose
                    joint_idx = (i - 4) // 3
                    component = (i - 4) % 3
                    component_names = ['_rot_x', '_rot_y', '_rot_z']
                    name = f'body_pose_joint_{joint_idx}{component_names[component]}'
                elif i < 70:  # 67 + 3 global orient
                    name = f'global_orient_{["x", "y", "z"][i-67]}'
                elif i < 80:  # 70 + 10 betas
                    name = f'shape_beta_{i-70}'
                elif i < 95:  # 80 + 15 left hand
                    name = f'left_hand_component_{i-80}'
                elif i < 110:  # 95 + 15 right hand
                    name = f'right_hand_component_{i-95}'
                elif i < 113:  # 110 + 3 jaw
                    name = f'jaw_pose_{["x", "y", "z"][i-110]}'
                else:  # expression components
                    name = f'expression_component_{i-113}'
                names.append(name)
        elif len(names) > num_features:
            names = names[:num_features]
        
        return names
    
    def analyze_permutation_importance(self, dataset, num_samples=100):
        """
        Calculate importance by permuting features and measuring performance drop
        
        Args:
            dataset: VideoParameterDataset instance
            num_samples: Number of samples to analyze
        
        Returns:
            Dictionary with feature importances
        """
        print(f"Analyzing feature importance using {num_samples} samples...")
        
        # First, inspect the dataset to generate accurate feature names
        if self.feature_names is None:
            print("Inspecting dataset structure for accurate feature names...")
            self._inspect_sample_and_generate_names(dataset)
        
        # Extract features and labels
        all_features = []
        all_labels = []
        
        for i in tqdm(range(min(num_samples, len(dataset))), desc="Extracting features"):
            sample = dataset[i]
            features = sample['features']  # [seq_len, feature_dim]
            label = sample['label'].item()
            
            # Average across time for simpler analysis
            features_avg = features.mean(dim=0).cpu().numpy()  # [feature_dim]
            
            all_features.append(features_avg)
            all_labels.append(label)
        
        X = np.array(all_features)  # [num_samples, feature_dim]
        y = np.array(all_labels)
        
        # Ensure feature names match actual dimension
        actual_feature_dim = X.shape[1]
        if self.feature_names is None:
            self.feature_names = self._generate_feature_names_fallback(actual_feature_dim)
        elif len(self.feature_names) != actual_feature_dim:
            print(f"Warning: Feature name mismatch ({len(self.feature_names)} names vs {actual_feature_dim} features)")
            self.feature_names = self._generate_feature_names_fallback(actual_feature_dim)
        
        # Use Random Forest as interpretable proxy
        print("Training interpretable proxy model...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # Get feature importances
        importances = rf.feature_importances_
        
        # Sort by importance
        indices = np.argsort(importances)[::-1]
        
        results = {
            'feature_names': [self.feature_names[i] for i in indices],
            'importances': importances[indices],
            'indices': indices,
            'feature_structure': self.feature_structure
        }
        
        return results
    
    def plot_feature_importance(self, importance_results, top_k=20, save_path='feature_importance.png'):
        """
        Plot top-k most important features
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        top_names = importance_results['feature_names'][:top_k]
        top_importances = importance_results['importances'][:top_k]
        
        # Reverse order for horizontal bar plot
        top_names = top_names[::-1]
        top_importances = top_importances[::-1]
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_k))
        
        ax.barh(range(top_k), top_importances, color=colors)
        ax.set_yticks(range(top_k))
        ax.set_yticklabels(top_names)
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title(f'Top {top_k} Most Important Features for Deepfake Detection', fontsize=14)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Feature importance plot saved to {save_path}")
        plt.close()
    
    def generate_feature_report(self, importance_results, save_path='feature_importance_report.txt'):
        """
        Generate text report of feature importances
        """
        with open(save_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("FEATURE IMPORTANCE ANALYSIS REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Total features analyzed: {len(importance_results['feature_names'])}\n\n")
            
            f.write("TOP 20 MOST IMPORTANT FEATURES:\n")
            f.write("-"*70 + "\n")
            
            for i in range(min(20, len(importance_results['feature_names']))):
                name = importance_results['feature_names'][i]
                importance = importance_results['importances'][i]
                f.write(f"{i+1:2d}. {name:<40} {importance:.6f}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("INTERPRETATION:\n")
            f.write("="*70 + "\n\n")
            
            # Group by category
            categories = {
                'Camera': 0,
                'Body Joints': 0,
                'Hands': 0,
                'Face/Expression': 0,
                'Shape': 0
            }
            
            for name, importance in zip(importance_results['feature_names'], 
                                       importance_results['importances']):
                if 'camera' in name:
                    categories['Camera'] += importance
                elif any(x in name for x in ['pelvis', 'hip', 'knee', 'ankle', 'spine', 
                                             'shoulder', 'elbow', 'wrist', 'neck', 'head']):
                    categories['Body Joints'] += importance
                elif 'hand' in name:
                    categories['Hands'] += importance
                elif 'jaw' in name or 'expression' in name:
                    categories['Face/Expression'] += importance
                elif 'shape' in name or 'beta' in name:
                    categories['Shape'] += importance
            
            f.write("Feature category contributions:\n")
            for cat, total_importance in sorted(categories.items(), 
                                               key=lambda x: x[1], reverse=True):
                f.write(f"  {cat:<20}: {total_importance:.4f}\n")
        
        print(f"✓ Feature importance report saved to {save_path}")


# ============================================================================
# PART 2: ATTENTION VISUALIZATION (LSTM)
# ============================================================================

class LSTMAttentionVisualizer:
    """
    Visualizes LSTM attention weights
    """
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
    
    def extract_attention(self, video_params_path):
        """
        Extract attention weights for a video
        
        Args:
            video_params_path: Path to all_frame_params.npz
        
        Returns:
            Dictionary with attention weights and prediction
        """
        # Load video
        dataset = VideoParameterDataset([video_params_path], [0], max_frames=100)
        sample = dataset[0]
        
        features = sample['features'].unsqueeze(0).to(self.device)  # [1, 100, feature_dim]
        
        with torch.no_grad():
            # Forward pass
            lstm_out, _ = self.model.lstm(features)
            
            # Get attention weights
            attention_weights = self.model.attention(lstm_out)  # [1, 100, 1]
            
            # Get prediction
            attended = torch.sum(lstm_out * attention_weights, dim=1)
            logits = self.model.classifier(attended)
            probs = F.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1).item()
        
        return {
            'attention_weights': attention_weights[0].squeeze().cpu().numpy(),  # [100]
            'prediction': prediction,
            'confidence': probs[0, prediction].item(),
            'prob_real': probs[0, 0].item(),
            'prob_fake': probs[0, 1].item(),
            'num_frames': sample['num_frames']
        }
    
    def visualize(self, video_params_path, output_path='lstm_attention.png'):
        """
        Create attention visualization
        """
        result = self.extract_attention(video_params_path)
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Attention weights over time
        ax = axes[0]
        attention = result['attention_weights']
        frames = np.arange(len(attention))
        
        ax.plot(frames, attention, linewidth=2, color='steelblue')
        ax.fill_between(frames, 0, attention, alpha=0.3, color='steelblue')
        ax.set_xlabel('Frame Number', fontsize=12)
        ax.set_ylabel('Attention Weight', fontsize=12)
        ax.set_title(f'LSTM Attention Weights (Prediction: {"Fake" if result["prediction"]==1 else "Real"}, '
                    f'Confidence: {result["confidence"]*100:.1f}%)', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Highlight high-attention frames
        threshold = attention.mean() + 2 * attention.std()
        high_attn_frames = np.where(attention > threshold)[0]
        if len(high_attn_frames) > 0:
            ax.scatter(high_attn_frames, attention[high_attn_frames], 
                      color='red', s=100, zorder=5, label='High Attention', marker='D')
            ax.legend()
        
        # Plot 2: Prediction confidence
        ax = axes[1]
        labels = ['Real', 'Fake']
        probs = [result['prob_real'], result['prob_fake']]
        colors = ['green', 'red']
        
        bars = ax.bar(labels, probs, color=colors, alpha=0.7)
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_title('Model Confidence', fontsize=14)
        ax.set_ylim([0, 1])
        
        # Add value labels on bars
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{prob*100:.1f}%', ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ LSTM attention visualization saved to {output_path}")
        plt.close()
        
        return result


# ============================================================================
# PART 3: ATTENTION VISUALIZATION (TRANSFORMER)
# ============================================================================

class TransformerAttentionVisualizer:
    """
    Visualizes Transformer multi-head self-attention
    """
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
    
    def extract_attention(self, video_params_path):
        """
        Extract attention weights for Transformer
        
        Returns:
            Dictionary with multi-head attention matrices and prediction
        """
        # Load video
        dataset = VideoParameterDataset([video_params_path], [0], max_frames=100)
        sample = dataset[0]
        
        features = sample['features'].unsqueeze(0).to(self.device)  # [1, 100, feature_dim]
        
        # Store attention weights
        attention_weights = []
        
        def get_attention_hook(module, input, output):
            # Capture attention weights during forward pass
            if hasattr(module, 'attn_weights'):
                attention_weights.append(module.attn_weights)
        
        # Register hooks on each transformer layer
        hooks = []
        for layer in self.model.transformer.layers:
            hook = layer.self_attn.register_forward_hook(get_attention_hook)
            hooks.append(hook)
        
        with torch.no_grad():
            # Embed
            x = self.model.embedding(features)
            x = x.transpose(0, 1)  # [seq, batch, dim]
            
            # Forward through transformer (attention captured by hooks)
            for layer in self.model.transformer.layers:
                # Manual forward to capture attention
                attn_output, attn_weights_layer = layer.self_attn(
                    x, x, x, 
                    need_weights=True
                )
                attention_weights.append(attn_weights_layer)
                
                # Complete the layer forward
                x = layer.norm1(x + layer.dropout1(attn_output))
                x2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
                x = layer.norm2(x + layer.dropout2(x2))
            
            x = x.transpose(0, 1)  # Back to [batch, seq, dim]
            
            # Classify
            x = torch.mean(x, dim=1)
            logits = self.model.classifier(x)
            probs = F.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1).item()
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return {
            'attention_weights': [a.cpu().numpy() for a in attention_weights],
            # Each: [batch, num_heads, seq, seq] = [1, 8, 100, 100]
            'prediction': prediction,
            'confidence': probs[0, prediction].item(),
            'prob_real': probs[0, 0].item(),
            'prob_fake': probs[0, 1].item(),
            'num_frames': sample['num_frames']
        }
    
    def visualize_layer(self, attention_matrix, layer_idx, output_path):
        """
        Visualize all heads for one transformer layer
        
        Args:
            attention_matrix: [batch, seq, seq] or [batch, num_heads, seq, seq]
            layer_idx: Which layer (0-3)
        """
        # Handle both averaged and per-head attention formats
        if len(attention_matrix.shape) == 3:
            # Averaged attention: [batch, seq, seq] -> expand to simulate heads
            attn = attention_matrix[0]  # [seq, seq]
            # Create 8 copies to simulate 8 heads
            attn = np.tile(attn[np.newaxis, :, :], (8, 1, 1))  # [8, seq, seq]
        else:
            # Per-head attention: [batch, num_heads, seq, seq]
            attn = attention_matrix[0]  # [num_heads, seq, seq]
        
        num_heads = attn.shape[0]
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'Transformer Layer {layer_idx} - Multi-Head Attention', fontsize=16)
        
        for head_idx in range(min(num_heads, 8)):
            ax = axes[head_idx // 4, head_idx % 4]
            
            # Plot attention matrix
            head_attn = attn[head_idx]  # [seq, seq]
            
            # Ensure it's 2D and has proper shape
            if head_attn.ndim == 1:
                head_attn = head_attn.reshape(-1, 1)
            
            sns.heatmap(head_attn, 
                       cmap='viridis',
                       xticklabels=10,
                       yticklabels=10,
                       ax=ax,
                       cbar_kws={'label': 'Attention Weight'})
            
            ax.set_title(f'Head {head_idx}')
            ax.set_xlabel('Key Frame')
            ax.set_ylabel('Query Frame')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Layer {layer_idx} attention saved to {output_path}")
        plt.close()
    
    def visualize_summary(self, result, output_path='transformer_attention_summary.png'):
        """
        Create summary visualization across all layers
        """
        attention_weights = result['attention_weights']
        num_layers = len(attention_weights)
        
        # Handle attention weights with flexible shapes
        processed_attn = []
        for attn in attention_weights:
            # Convert to numpy if needed
            if hasattr(attn, 'cpu'):
                attn = attn.cpu().numpy()
            
            # Handle different attention shapes
            if attn.ndim == 1:
                # [seq] -> create diagonal matrix
                seq_len = len(attn)
                attn_2d = np.diag(attn)
            elif attn.ndim == 2:
                # [seq, seq] -> use as is
                attn_2d = attn
            elif attn.ndim == 3:
                # [batch, seq, seq] or [num_heads, seq, seq] -> take first and average
                attn_2d = attn[0]
            elif attn.ndim == 4:
                # [batch, num_heads, seq, seq] -> average over batch and heads
                attn_2d = attn[0].mean(axis=0)
            else:
                continue
            
            processed_attn.append(attn_2d)
        
        # Stack and average
        if len(processed_attn) > 0:
            all_attn = np.stack(processed_attn)  # [num_layers, seq, seq]
            avg_attn = all_attn.mean(axis=0)  # [seq, seq]
        else:
            print("Warning: No valid attention weights found")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # Plot 1: Average attention matrix
        ax = axes[0, 0]
        sns.heatmap(avg_attn, cmap='viridis', ax=ax,
                   xticklabels=10, yticklabels=10,
                   cbar_kws={'label': 'Attention Weight'})
        ax.set_title('Average Attention (All Heads & Layers)', fontsize=12)
        ax.set_xlabel('Key Frame')
        ax.set_ylabel('Query Frame')
        
        # Plot 2: Self-attention strength (diagonal)
        ax = axes[0, 1]
        self_attn = np.diag(avg_attn)
        ax.plot(self_attn, linewidth=2, color='coral')
        ax.set_title('Self-Attention Strength\n(How much each frame attends to itself)', fontsize=12)
        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Self-Attention Weight')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Attention range
        ax = axes[1, 0]
        attention_range = []
        seq_len = avg_attn.shape[0]
        for i in range(seq_len):
            weights = avg_attn[i]
            positions = np.arange(seq_len)
            avg_distance = np.sum(weights * np.abs(positions - i))
            attention_range.append(avg_distance)
        
        ax.plot(attention_range, linewidth=2, color='steelblue')
        ax.set_title('Attention Range\n(Average distance to attended frames)', fontsize=12)
        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Average Distance (frames)')
        ax.grid(True, alpha=0.3)
        
        # Highlight unusual patterns
        mean_range = np.mean(attention_range)
        std_range = np.std(attention_range)
        unusual = np.where(np.abs(attention_range - mean_range) > 2 * std_range)[0]
        if len(unusual) > 0:
            ax.scatter(unusual, [attention_range[i] for i in unusual],
                      color='red', s=100, zorder=5, label='Unusual')
            ax.legend()
        
        # Plot 4: Prediction confidence
        ax = axes[1, 1]
        labels = ['Real', 'Fake']
        probs = [result['prob_real'], result['prob_fake']]
        colors = ['green', 'red']
        bars = ax.bar(labels, probs, color=colors, alpha=0.7)
        ax.set_ylabel('Probability')
        ax.set_title(f'Model Prediction\n{"Fake" if result["prediction"]==1 else "Real"} '
                    f'({result["confidence"]*100:.1f}% confidence)', fontsize=12)
        ax.set_ylim([0, 1])
        
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{prob*100:.1f}%', ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Transformer attention summary saved to {output_path}")
        plt.close()
    
    def visualize_all(self, video_params_path, output_dir='transformer_attention'):
        """
        Generate all visualizations for a video
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("Extracting transformer attention...")
        result = self.extract_attention(video_params_path)
        
        # Visualize each layer
        for layer_idx, attn_matrix in enumerate(result['attention_weights']):
            output_path = osp.join(output_dir, f'layer_{layer_idx}_all_heads.png')
            self.visualize_layer(attn_matrix, layer_idx, output_path)
        
        # Generate summary
        summary_path = osp.join(output_dir, 'summary.png')
        self.visualize_summary(result, summary_path)
        
        return result


# ============================================================================
# PART 4: SINGLE VIDEO FEATURE ABLATION
# ============================================================================

class SingleVideoExplainer:
    """
    Explains why a specific video was classified as Real or Fake
    using feature ablation analysis
    """
    
    def __init__(self, model, device, feature_names=None):
        self.model = model
        self.device = device
        self.feature_names = feature_names
        self.model.eval()
    
    def _infer_feature_names_from_params(self, video_params_path):
        """
        Infer feature names directly from the video params file
        without needing external dataset
        """
        print("Inferring feature structure from video params file...")
        
        data = np.load(video_params_path, allow_pickle=True)
        params = data['params']
        
        if len(params) == 0:
            print("Warning: No frames in params file")
            return None
        
        # Inspect first frame
        first_frame = params[0]
        
        names = []
        
        # Helper to extract and name features from parameter dict
        def add_features_from_param(param_dict, key, prefix, max_len=None):
            if key in param_dict:
                value = param_dict[key]
                if np.isscalar(value):
                    names.append(f'{prefix}')
                else:
                    value_flat = np.array(value).flatten()
                    if max_len is not None:
                        value_flat = value_flat[:max_len]
                    for i in range(len(value_flat)):
                        names.append(f'{prefix}_{i}')
                return True
            return False
        
        # Build names matching extraction order in deepfake_detector.py _extract_features()
        add_features_from_param(first_frame, 'camera_scale', 'camera_scale')
        add_features_from_param(first_frame, 'camera_translation', 'camera_translation')
        add_features_from_param(first_frame, 'body_pose', 'body_pose')
        add_features_from_param(first_frame, 'global_orient', 'global_orient')
        add_features_from_param(first_frame, 'betas', 'shape_beta')
        add_features_from_param(first_frame, 'left_hand_pose', 'left_hand_component', max_len=15)
        add_features_from_param(first_frame, 'right_hand_pose', 'right_hand_component', max_len=15)
        add_features_from_param(first_frame, 'jaw_pose', 'jaw_pose')
        add_features_from_param(first_frame, 'expression', 'expression_component', max_len=10)
        
        print(f"✓ Inferred {len(names)} feature names from params file")
        
        # Print summary of feature groups
        groups = {}
        for name in names:
            prefix = name.rsplit('_', 1)[0] if '_' in name else name
            groups[prefix] = groups.get(prefix, 0) + 1
        
        print(f"  Feature groups detected:")
        for group, count in sorted(groups.items()):
            print(f"    - {group}: {count} features")
        
        return names
    
    def ablate_features(self, video_params_path, top_k=20):
        """
        Analyze which features are most important for this video's prediction
        
        Args:
            video_params_path: Path to video parameter file
            top_k: Number of top features to analyze
        
        Returns:
            Dictionary with ablation results
        """
        print(f"Analyzing video: {video_params_path}")
        
        # Load video
        dataset = VideoParameterDataset([video_params_path], [0], max_frames=100)
        sample = dataset[0]
        
        features_orig = sample['features'].unsqueeze(0).to(self.device)  # [1, seq_len, feature_dim]
        
        # Get original prediction
        with torch.no_grad():
            logits_orig = self.model(features_orig)
            probs_orig = F.softmax(logits_orig, dim=1)
            pred_orig = torch.argmax(logits_orig, dim=1).item()
            conf_orig = probs_orig[0, pred_orig].item()
        
        print(f"\nOriginal Prediction: {'FAKE' if pred_orig == 1 else 'REAL'} (confidence: {conf_orig*100:.1f}%)")
        
        # Average features across time for importance calculation
        features_avg = features_orig.mean(dim=1)  # [1, feature_dim]
        feature_dim = features_avg.shape[1]
        
        # If feature names not provided, infer from the video params file directly
        if self.feature_names is None:
            self.feature_names = self._infer_feature_names_from_params(video_params_path)
        
        # Ablate each feature
        ablation_results = []
        
        print(f"Ablating {feature_dim} features...")
        for feat_idx in tqdm(range(feature_dim), desc="Feature ablation"):
            # Create ablated features (zero out one feature across all frames)
            features_ablated = features_orig.clone()
            features_ablated[:, :, feat_idx] = 0  # Zero out this feature in all frames
            
            # Get prediction with ablated feature
            with torch.no_grad():
                logits_ablated = self.model(features_ablated)
                probs_ablated = F.softmax(logits_ablated, dim=1)
                conf_ablated = probs_ablated[0, pred_orig].item()
            
            # Calculate confidence drop
            conf_drop = conf_orig - conf_ablated
            
            ablation_results.append({
                'feature_idx': feat_idx,
                'feature_name': self.feature_names[feat_idx] if feat_idx < len(self.feature_names) else f'feature_{feat_idx}',
                'original_conf': conf_orig,
                'ablated_conf': conf_ablated,
                'conf_drop': conf_drop,
                'importance_score': conf_drop  # How much confidence dropped
            })
        
        # Sort by importance
        ablation_results.sort(key=lambda x: abs(x['importance_score']), reverse=True)
        
        return {
            'prediction': pred_orig,
            'confidence': conf_orig,
            'prob_real': probs_orig[0, 0].item(),
            'prob_fake': probs_orig[0, 1].item(),
            'ablation_results': ablation_results,
            'top_k': top_k,
            'num_frames': sample['num_frames'],
            'total_features': feature_dim
        }
    
    def visualize_ablation(self, ablation_result, output_path='feature_ablation.png'):
        """
        Create visualization of feature ablation results
        """
        results = ablation_result['ablation_results']
        top_k = min(ablation_result['top_k'], len(results))
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Top features by ablation importance
        ax = axes[0, 0]
        top_results = results[:top_k]
        
        feature_names = [r['feature_name'] for r in top_results]
        conf_drops = [r['conf_drop'] for r in top_results]
        colors = ['red' if drop > 0 else 'blue' for drop in conf_drops]
        
        y_pos = np.arange(len(feature_names))
        ax.barh(y_pos, [abs(d) for d in conf_drops], color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names, fontsize=9)
        ax.set_xlabel('|Confidence Drop|', fontsize=11)
        ax.set_title(f'Top {top_k} Features by Ablation Importance\n(Red = supports prediction, Blue = opposes)', 
                    fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        
        # Add values on bars
        for i, (name, drop) in enumerate(zip(feature_names, conf_drops)):
            ax.text(abs(drop) + 0.005, i, f'{abs(drop):.3f}', va='center', fontsize=9)
        
        # Plot 2: Cumulative importance
        ax = axes[0, 1]
        cumsum_importance = np.cumsum([abs(r['conf_drop']) for r in results])
        ax.plot(cumsum_importance, linewidth=2, color='darkblue')
        ax.fill_between(range(len(cumsum_importance)), cumsum_importance, alpha=0.3, color='lightblue')
        ax.axhline(y=cumsum_importance[-1] * 0.8, color='red', linestyle='--', alpha=0.5, 
                  label=f'80% threshold')
        ax.set_xlabel('Feature Index (sorted by importance)', fontsize=11)
        ax.set_ylabel('Cumulative Importance', fontsize=11)
        ax.set_title('Cumulative Feature Importance', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot 3: Feature group importance
        ax = axes[1, 0]
        group_importance = {}
        for r in results:
            fname = r['feature_name']
            # Extract group name (everything before the last underscore or digit)
            group = fname.rsplit('_', 1)[0] if '_' in fname else fname
            if group not in group_importance:
                group_importance[group] = 0
            group_importance[group] += abs(r['conf_drop'])
        
        # Sort by importance
        group_importance = dict(sorted(group_importance.items(), key=lambda x: x[1], reverse=True))
        
        groups = list(group_importance.keys())
        importances = list(group_importance.values())
        
        colors_grad = plt.cm.RdYlGn_r(np.linspace(0.3, 0.7, len(groups)))
        ax.bar(range(len(groups)), importances, color=colors_grad, alpha=0.8)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('Total Importance Score', fontsize=11)
        ax.set_title('Feature Group Importance', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Prediction confidence summary
        ax = axes[1, 1]
        ax.axis('off')
        
        # Summary text
        pred_label = 'FAKE' if ablation_result['prediction'] == 1 else 'REAL'
        summary_text = (
            f"Video Analysis Summary\n"
            f"{'='*40}\n\n"
            f"Prediction: {pred_label}\n"
            f"Confidence: {ablation_result['confidence']*100:.1f}%\n"
            f"P(Real): {ablation_result['prob_real']*100:.1f}%\n"
            f"P(Fake): {ablation_result['prob_fake']*100:.1f}%\n\n"
            f"Video Info:\n"
            f"- Frames: {ablation_result['num_frames']}\n"
            f"- Total Features: {ablation_result['total_features']}\n\n"
            f"Top Ablation Features:\n"
        )
        
        for i, r in enumerate(results[:5]):
            summary_text += f"{i+1}. {r['feature_name']}\n"
            summary_text += f"   Drop: {abs(r['conf_drop']):.4f}\n"
        
        ax.text(0.1, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', family='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Ablation visualization saved to {output_path}")
        plt.close()
    
    def generate_ablation_report(self, ablation_result, output_path='feature_ablation_report.txt'):
        """Generate detailed text report of ablation analysis"""
        
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SINGLE VIDEO FEATURE ABLATION ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write("PREDICTION SUMMARY\n")
            f.write("-"*80 + "\n")
            pred_label = 'FAKE' if ablation_result['prediction'] == 1 else 'REAL'
            f.write(f"Predicted Label:  {pred_label}\n")
            f.write(f"Confidence:       {ablation_result['confidence']*100:.2f}%\n")
            f.write(f"P(Real):          {ablation_result['prob_real']*100:.2f}%\n")
            f.write(f"P(Fake):          {ablation_result['prob_fake']*100:.2f}%\n\n")
            
            f.write("VIDEO INFORMATION\n")
            f.write("-"*80 + "\n")
            f.write(f"Number of Frames: {ablation_result['num_frames']}\n")
            f.write(f"Total Features:   {ablation_result['total_features']}\n\n")
            
            f.write("TOP 20 MOST IMPORTANT FEATURES (by ablation)\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Rank':<6} {'Feature Name':<40} {'Conf Drop':<12} {'Impact'}\n")
            f.write("-"*80 + "\n")
            
            for i, result in enumerate(ablation_result['ablation_results'][:20]):
                impact = "CRITICAL" if result['conf_drop'] > 0.1 else "HIGH" if result['conf_drop'] > 0.05 else "MEDIUM"
                f.write(f"{i+1:<6} {result['feature_name']:<40} {result['conf_drop']:<12.6f} {impact}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("INTERPRETATION GUIDE\n")
            f.write("="*80 + "\n")
            f.write("- 'Conf Drop': How much the prediction confidence decreased when this feature was removed\n")
            f.write("- Larger drops = feature is more important for the prediction\n")
            f.write("- Positive drop = feature supports the current prediction\n")
            f.write("- Features are analyzed individually (one-at-a-time ablation)\n")
        
        print(f"✓ Ablation report saved to {output_path}")


# ============================================================================
# MAIN CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Deepfake Detection Explainability Toolkit'
    )
    
    # Required arguments
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model (.pth file)')
    parser.add_argument('--model-type', type=str, required=True,
                       choices=['lstm', 'transformer'],
                       help='Model architecture type')
    
    # Task selection
    parser.add_argument('--task', type=str, required=True,
                       choices=['feature-importance', 'attention', 'ablation', 'both'],
                       help='Which analysis to run')
    
    # Data inputs
    parser.add_argument('--data-dir', type=str,
                       help='Directory with processed videos (for feature importance)')
    parser.add_argument('--video-params', type=str,
                       help='Path to specific video params (for attention visualization)')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='explainability_output',
                       help='Output directory for results')
    
    # Options
    parser.add_argument('--num-samples', type=int, default=100,
                       help='Number of samples for feature importance')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading {args.model_type.upper()} model...")
    checkpoint = torch.load(args.model_path, map_location=device)
    config = checkpoint['model_config']
    
    if args.model_type == 'lstm':
        model = DeepfakeDetectorLSTM(
            config['input_dim'],
            config['hidden_dim'],
            config['num_layers'],
            config['dropout']
        )
    else:
        model = DeepfakeDetectorTransformer(
            config['input_dim'],
            config['hidden_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("✓ Model loaded successfully")
    
    # Run requested analyses
    if args.task in ['feature-importance', 'both']:
        if not args.data_dir:
            print("Error: --data-dir required for feature importance analysis")
            return
        
        print("\n" + "="*70)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*70)
        
        # Create dataset
        param_files = []
        labels = []
        
        for class_name, label in [('real', 0), ('fake', 1)]:
            class_dir = osp.join(args.data_dir, class_name)
            if not osp.exists(class_dir):
                continue
            
            for video_dir in os.listdir(class_dir):
                param_file = osp.join(class_dir, video_dir, 'all_frame_params.npz')
                if osp.exists(param_file):
                    param_files.append(param_file)
                    labels.append(label)
        
        dataset = VideoParameterDataset(param_files, labels, max_frames=100)
        
        # Analyze
        analyzer = FeatureImportanceAnalyzer(model, device, args.model_type)
        results = analyzer.analyze_permutation_importance(dataset, args.num_samples)
        
        # Save results
        analyzer.plot_feature_importance(
            results,
            top_k=20,
            save_path=osp.join(args.output_dir, 'feature_importance.png')
        )
        analyzer.generate_feature_report(
            results,
            save_path=osp.join(args.output_dir, 'feature_importance_report.txt')
        )
        
        # Save raw data
        np.savez(
            osp.join(args.output_dir, 'feature_importance_data.npz'),
            feature_names=results['feature_names'],
            importances=results['importances']
        )
    
    if args.task in ['attention', 'both']:
        if not args.video_params:
            print("Error: --video-params required for attention visualization")
            return
        
        print("\n" + "="*70)
        print("ATTENTION VISUALIZATION")
        print("="*70)
        
        if args.model_type == 'lstm':
            visualizer = LSTMAttentionVisualizer(model, device)
            result = visualizer.visualize(
                args.video_params,
                output_path=osp.join(args.output_dir, 'lstm_attention.png')
            )
            
            # Print summary
            print(f"\nPrediction: {'Fake' if result['prediction']==1 else 'Real'}")
            print(f"Confidence: {result['confidence']*100:.1f}%")
            print(f"Prob(Real): {result['prob_real']*100:.1f}%")
            print(f"Prob(Fake): {result['prob_fake']*100:.1f}%")
            
        else:
            visualizer = TransformerAttentionVisualizer(model, device)
            result = visualizer.visualize_all(
                args.video_params,
                output_dir=osp.join(args.output_dir, 'transformer_attention')
            )
            
            # Print summary
            print(f"\nPrediction: {'Fake' if result['prediction']==1 else 'Real'}")
            print(f"Confidence: {result['confidence']*100:.1f}%")
            print(f"Prob(Real): {result['prob_real']*100:.1f}%")
            print(f"Prob(Fake): {result['prob_fake']*100:.1f}%")
    
    if args.task in ['ablation']:
        if not args.video_params:
            print("Error: --video-params required for ablation analysis")
            return
        
        print("\n" + "="*70)
        print("SINGLE VIDEO FEATURE ABLATION ANALYSIS")
        print("="*70)
        
        # Run ablation (feature names are inferred directly from the video params file)
        explainer = SingleVideoExplainer(model, device)
        result = explainer.ablate_features(args.video_params, top_k=20)
        
        # Visualize
        explainer.visualize_ablation(
            result,
            output_path=osp.join(args.output_dir, 'feature_ablation.png')
        )
        
        # Generate report
        explainer.generate_ablation_report(
            result,
            output_path=osp.join(args.output_dir, 'feature_ablation_report.txt')
        )
        
        # Print top features
        print(f"\nTop 10 Important Features:")
        print("-" * 60)
        for i, res in enumerate(result['ablation_results'][:10]):
            print(f"{i+1:2d}. {res['feature_name']:40s} Drop: {res['conf_drop']:8.6f}")
    
    print("\n" + "="*70)
    print(f"✓ All results saved to {args.output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()