#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model Architecture Visualization Script

Generates visual representations of the LSTM and Transformer architectures
used in the deepfake detection system.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

def visualize_lstm_architecture(output_path='model_architecture_lstm.png'):
    """Create visualization of LSTM-based deepfake detector"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, 'LSTM-Based Deepfake Detector Architecture', 
            fontsize=18, fontweight='bold', ha='center')
    
    # Color scheme
    color_input = '#E8F4F8'
    color_process = '#B3E5FC'
    color_attention = '#FFE0B2'
    color_output = '#C8E6C9'
    color_edge = '#424242'
    
    y_pos = 10.5
    x_center = 5
    box_width = 2
    box_height = 0.6
    
    # 1. Input Layer
    box1 = FancyBboxPatch((x_center-box_width/2, y_pos-box_height/2), box_width, box_height,
                          boxstyle="round,pad=0.1", edgecolor=color_edge, facecolor=color_input, linewidth=2)
    ax.add_patch(box1)
    ax.text(x_center, y_pos, 'Input: Video Frames\n(batch, seq_len, feature_dim)', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrow
    ax.arrow(x_center, y_pos-0.5, 0, -0.8, head_width=0.2, head_length=0.15, fc=color_edge, ec=color_edge)
    
    # 2. LSTM Layers
    y_pos = 8.8
    box2 = FancyBboxPatch((x_center-box_width/2, y_pos-box_height/2), box_width, box_height,
                          boxstyle="round,pad=0.1", edgecolor=color_edge, facecolor=color_process, linewidth=2)
    ax.add_patch(box2)
    ax.text(x_center, y_pos, 'LSTM (Bidirectional)\n2 layers × 256 hidden units', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Add side annotations for LSTM
    ax.text(x_center + 2.5, y_pos + 0.3, '• Processes temporal sequence\n• Captures long-term dependencies\n• Output: (batch, seq_len, 512)', 
            fontsize=9, va='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Arrow
    ax.arrow(x_center, y_pos-0.5, 0, -0.8, head_width=0.2, head_length=0.15, fc=color_edge, ec=color_edge)
    
    # 3. Attention Layer
    y_pos = 7.1
    box3 = FancyBboxPatch((x_center-box_width/2, y_pos-box_height/2), box_width, box_height,
                          boxstyle="round,pad=0.1", edgecolor=color_edge, facecolor=color_attention, linewidth=2)
    ax.add_patch(box3)
    ax.text(x_center, y_pos, 'Attention Mechanism\n512 → 128 → 1 (softmax)', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Add side annotations for attention
    ax.text(x_center + 2.5, y_pos + 0.3, '• Learns temporal weights\n• Emphasizes important frames\n• Weighted sum of LSTM outputs', 
            fontsize=9, va='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Arrow
    ax.arrow(x_center, y_pos-0.5, 0, -0.8, head_width=0.2, head_length=0.15, fc=color_edge, ec=color_edge)
    
    # 4. Attended Feature Vector
    y_pos = 5.4
    box4 = FancyBboxPatch((x_center-box_width/2, y_pos-box_height/2), box_width, box_height,
                          boxstyle="round,pad=0.1", edgecolor=color_edge, facecolor=color_process, linewidth=2)
    ax.add_patch(box4)
    ax.text(x_center, y_pos, 'Attended Features\n(batch, 512)', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrow
    ax.arrow(x_center, y_pos-0.5, 0, -0.8, head_width=0.2, head_length=0.15, fc=color_edge, ec=color_edge)
    
    # 5. Classifier
    y_pos = 3.7
    box5 = FancyBboxPatch((x_center-box_width/2, y_pos-box_height/2), box_width, box_height,
                          boxstyle="round,pad=0.1", edgecolor=color_edge, facecolor=color_process, linewidth=2)
    ax.add_patch(box5)
    ax.text(x_center, y_pos, 'Classifier\n512 → 128 → 64 → 2', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Add side annotations for classifier
    ax.text(x_center + 2.5, y_pos + 0.3, '• 3 fully connected layers\n• ReLU activations + Dropout\n• Output logits for 2 classes', 
            fontsize=9, va='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Arrow
    ax.arrow(x_center, y_pos-0.5, 0, -0.8, head_width=0.2, head_length=0.15, fc=color_edge, ec=color_edge)
    
    # 6. Output Layer
    y_pos = 2.0
    box6 = FancyBboxPatch((x_center-box_width/2, y_pos-box_height/2), box_width, box_height,
                          boxstyle="round,pad=0.1", edgecolor=color_edge, facecolor=color_output, linewidth=2)
    ax.add_patch(box6)
    ax.text(x_center, y_pos, 'Output: Softmax\n[Real Probability, Fake Probability]', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Add legend/info
    info_text = (
        'Key Parameters:\n'
        '• Input Features: SMPL-X parameters (119 dims)\n'
        '• Max Sequence Length: 100 frames\n'
        '• LSTM Hidden Dim: 256 (bidirectional → 512)\n'
        '• Dropout Rate: 0.3\n'
        '• Total Parameters: ~780K'
    )
    ax.text(0.5, 0.5, info_text, fontsize=9, 
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
            family='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ LSTM architecture saved to {output_path}")
    plt.close()


def visualize_transformer_architecture(output_path='model_architecture_transformer.png'):
    """Create visualization of Transformer-based deepfake detector"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13)
    ax.axis('off')
    
    # Title
    ax.text(5, 12.5, 'Transformer-Based Deepfake Detector Architecture', 
            fontsize=18, fontweight='bold', ha='center')
    
    # Color scheme
    color_input = '#E8F4F8'
    color_embedding = '#FFE0B2'
    color_transformer = '#F8BBD0'
    color_attention = '#E1BEE7'
    color_output = '#C8E6C9'
    color_edge = '#424242'
    
    y_pos = 11.8
    x_center = 5
    box_width = 2.2
    box_height = 0.6
    
    # 1. Input Layer
    box1 = FancyBboxPatch((x_center-box_width/2, y_pos-box_height/2), box_width, box_height,
                          boxstyle="round,pad=0.1", edgecolor=color_edge, facecolor=color_input, linewidth=2)
    ax.add_patch(box1)
    ax.text(x_center, y_pos, 'Input: Video Frames\n(batch, seq_len, feature_dim)', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax.arrow(x_center, y_pos-0.5, 0, -0.8, head_width=0.2, head_length=0.15, fc=color_edge, ec=color_edge)
    
    # 2. Embedding Layer
    y_pos = 10.1
    box2 = FancyBboxPatch((x_center-box_width/2, y_pos-box_height/2), box_width, box_height,
                          boxstyle="round,pad=0.1", edgecolor=color_edge, facecolor=color_embedding, linewidth=2)
    ax.add_patch(box2)
    ax.text(x_center, y_pos, 'Linear Embedding\nfeature_dim → 256', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax.text(x_center - 3.5, y_pos, 'Project input\nto d_model=256', 
            fontsize=9, va='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.arrow(x_center, y_pos-0.5, 0, -0.8, head_width=0.2, head_length=0.15, fc=color_edge, ec=color_edge)
    
    # 3. Transformer Encoder Stack
    y_pos = 8.0
    
    # Draw 4 encoder layers
    for layer in range(4):
        layer_y = 8.4 - layer * 1.3
        box = FancyBboxPatch((x_center-box_width/2, layer_y-box_height/2), box_width, box_height,
                             boxstyle="round,pad=0.1", edgecolor=color_edge, 
                             facecolor=color_transformer if layer < 3 else color_output, linewidth=2)
        ax.add_patch(box)
        
        if layer < 3:
            ax.text(x_center, layer_y, f'Encoder Layer {layer+1}', 
                    ha='center', va='center', fontsize=10, fontweight='bold')
        else:
            ax.text(x_center, layer_y, 'Encoder Layer 4\n(Final)', 
                    ha='center', va='center', fontsize=10, fontweight='bold')
        
        if layer < 3:
            ax.arrow(x_center, layer_y-0.5, 0, -0.5, head_width=0.15, head_length=0.1, 
                    fc=color_edge, ec=color_edge)
    
    # Add side annotation for encoder layers
    ax.text(x_center + 3.2, 6.8, (
        'Each Encoder Layer contains:\n'
        '• Multi-Head Attention (8 heads)\n'
        '• Feed-Forward Network\n'
        '• Layer Normalization\n'
        '• Residual Connections\n'
        '• Dropout (0.3)\n\n'
        'd_model: 256\n'
        'FF dimension: 1024\n'
        'Heads: 8\n'
        'Layers: 4'
    ), fontsize=9, va='center', 
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), family='monospace')
    
    # Arrow from encoder to pooling
    ax.arrow(x_center, 2.25, 0, -0.6, head_width=0.2, head_length=0.15, fc=color_edge, ec=color_edge)
    
    # 4. Global Average Pooling
    y_pos = 1.35
    box_pool = FancyBboxPatch((x_center-box_width/2, y_pos-box_height/2), box_width, box_height,
                              boxstyle="round,pad=0.1", edgecolor=color_edge, 
                              facecolor=color_embedding, linewidth=2)
    ax.add_patch(box_pool)
    ax.text(x_center, y_pos, 'Global Average Pooling\n(batch, 256)', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrow
    ax.arrow(x_center, y_pos-0.5, 0, -0.8, head_width=0.2, head_length=0.15, fc=color_edge, ec=color_edge)
    
    # 5. Classifier
    y_pos = -0.5
    box_clf = FancyBboxPatch((x_center-box_width/2, y_pos-box_height/2), box_width, box_height,
                             boxstyle="round,pad=0.1", edgecolor=color_edge, 
                             facecolor=color_output, linewidth=2)
    ax.add_patch(box_clf)
    ax.text(x_center, y_pos, 'Classifier\n256 → 128 → 2', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax.text(x_center - 3.5, y_pos, 'Final prediction\nHead (Binary)', 
            fontsize=9, va='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Arrow
    ax.arrow(x_center, y_pos-0.5, 0, -0.8, head_width=0.2, head_length=0.15, fc=color_edge, ec=color_edge)
    
    # 6. Output
    y_pos = -1.6
    ax.text(x_center, y_pos, 'Output: [Real Probability, Fake Probability]', 
            ha='center', va='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=color_output, alpha=0.8, linewidth=2))
    
    # Add legend/info
    info_text = (
        'Key Parameters:\n'
        '• Input Features: SMPL-X parameters (119 dims)\n'
        '• Model Dimension (d_model): 256\n'
        '• Attention Heads: 8\n'
        '• Encoder Layers: 4\n'
        '• FF Dimension: 1024\n'
        '• Max Sequence: 100 frames\n'
        '• Dropout: 0.3\n'
        '• Total Parameters: ~2.1M'
    )
    ax.text(0.3, -2.5, info_text, fontsize=9, 
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
            family='monospace', va='top')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Transformer architecture saved to {output_path}")
    plt.close()


def visualize_comparison(output_path='model_architecture_comparison.png'):
    """Create side-by-side comparison of both architectures"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    
    # Title
    fig.suptitle('Deepfake Detection: LSTM vs Transformer Architecture Comparison', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # ===== LEFT SIDE: LSTM =====
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 11)
    ax1.axis('off')
    ax1.text(5, 10.5, 'LSTM Architecture', fontsize=14, fontweight='bold', ha='center')
    
    color_edge = '#424242'
    
    # LSTM flow
    lstm_y_positions = [9.5, 8.2, 6.9, 5.6, 4.3, 3.0]
    lstm_labels = [
        'Input\n(seq_len, features)',
        'Bidirectional LSTM\n2 layers, 256 hidden',
        'Attention\nMechanism',
        'Attention Weights\n(softmax)',
        'Classifier\n3 FC layers',
        'Output\n2 classes'
    ]
    
    colors_lstm = ['#E8F4F8', '#B3E5FC', '#FFE0B2', '#FFE0B2', '#B3E5FC', '#C8E6C9']
    
    for i, (y, label, color) in enumerate(zip(lstm_y_positions, lstm_labels, colors_lstm)):
        box = FancyBboxPatch((2.5, y-0.35), 5, 0.7,
                             boxstyle="round,pad=0.05", edgecolor=color_edge, 
                             facecolor=color, linewidth=2)
        ax1.add_patch(box)
        ax1.text(5, y, label, ha='center', va='center', fontsize=9, fontweight='bold')
        
        if i < len(lstm_y_positions) - 1:
            ax1.arrow(5, y-0.4, 0, -0.4, head_width=0.15, head_length=0.1, fc=color_edge, ec=color_edge)
    
    # LSTM info
    lstm_info = (
        'Strengths:\n✓ Good for sequential data\n✓ Lighter weight\n✓ Fast training\n'
        '✓ ~780K parameters\n\n'
        'Weaknesses:\n✗ Limited long-range deps\n✗ Vanishing gradient risk'
    )
    ax1.text(5, 0.8, lstm_info, fontsize=8, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # ===== RIGHT SIDE: TRANSFORMER =====
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 11)
    ax2.axis('off')
    ax2.text(5, 10.5, 'Transformer Architecture', fontsize=14, fontweight='bold', ha='center')
    
    # Transformer flow (simplified)
    trans_y_positions = [9.5, 8.2, 6.5, 5.0, 3.5, 2.0]
    trans_labels = [
        'Input + Embedding\n(batch, seq, 256)',
        'Multi-Head Attention\n(8 heads, 4 layers)',
        'Feed-Forward Network\n(4× expansion)',
        'Layer Normalization\n& Residual Connections',
        'Global Avg Pooling\n(batch, 256)',
        'Output\n2 classes'
    ]
    
    colors_trans = ['#E8F4F8', '#F8BBD0', '#E1BEE7', '#F8BBD0', '#FFE0B2', '#C8E6C9']
    
    for i, (y, label, color) in enumerate(zip(trans_y_positions, trans_labels, colors_trans)):
        box = FancyBboxPatch((2.5, y-0.35), 5, 0.7,
                             boxstyle="round,pad=0.05", edgecolor=color_edge, 
                             facecolor=color, linewidth=2)
        ax2.add_patch(box)
        ax2.text(5, y, label, ha='center', va='center', fontsize=9, fontweight='bold')
        
        if i < len(trans_y_positions) - 1:
            ax2.arrow(5, y-0.4, 0, -0.4, head_width=0.15, head_length=0.1, fc=color_edge, ec=color_edge)
    
    # Transformer info
    trans_info = (
        'Strengths:\n✓ Excellent long-range deps\n✓ Parallel processing\n✓ Better explanability\n'
        '✓ ~2.1M parameters\n\n'
        'Weaknesses:\n✗ More compute needed\n✗ Slower inference'
    )
    ax2.text(5, 0.8, trans_info, fontsize=8, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='#FFE8E8', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Architecture comparison saved to {output_path}")
    plt.close()


def main():
    """Generate all architecture visualizations"""
    
    # Create output directory if it doesn't exist
    output_dir = 'model_architecture'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating model architecture visualizations...")
    print()
    
    # Generate visualizations
    visualize_lstm_architecture(os.path.join(output_dir, 'model_architecture_lstm.png'))
    visualize_transformer_architecture(os.path.join(output_dir, 'model_architecture_transformer.png'))
    visualize_comparison(os.path.join(output_dir, 'model_architecture_comparison.png'))
    
    print()
    print("✓ All architecture visualizations generated successfully!")
    print(f"✓ Output folder: {output_dir}/")
    print()
    print("Generated files:")
    print("  1. model_architecture_lstm.png - Detailed LSTM architecture")
    print("  2. model_architecture_transformer.png - Detailed Transformer architecture")
    print("  3. model_architecture_comparison.png - Side-by-side comparison")


if __name__ == '__main__':
    main()
