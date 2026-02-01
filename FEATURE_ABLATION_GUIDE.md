# Feature Ablation Analysis - Usage Guide

## Overview

Feature ablation analysis explains **why** a specific video was classified as Real or Fake by showing which features were most critical to the prediction.

## How It Works

1. **Extract features** from the video
2. **Get original prediction** (e.g., 95% confidence it's FAKE)
3. **Ablate each feature** one at a time (set it to zero)
4. **Measure confidence drop** when each feature is removed
5. **Rank features** by how much confidence dropped

**High confidence drop = critical feature for the prediction**

## Command Line Usage

### Basic Ablation Analysis (Standalone)

Ablation is now **completely standalone** - it infers feature names directly from the video params file:

```bash
python explainability_toolkit.py \
  --model-path path/to/model.pth \
  --model-type transformer \
  --task ablation \
  --video-params path/to/video_params.npz \
  --output-dir ./ablation_results
```

**No `--data-dir` needed!** The feature structure is inferred directly from the video params file.

## Output Files

### 1. `feature_ablation.png`

**4-panel visualization:**

**Panel 1 (Top-Left): Top 20 Features by Importance**
- Shows which features had the highest confidence drop when ablated
- Red bars = features support the prediction
- Blue bars = features oppose the prediction
- Height = magnitude of importance

**Panel 2 (Top-Right): Cumulative Feature Importance**
- Shows how many features are needed to explain 80% of the prediction
- Helps identify if prediction relies on few features (spiky curve) or many (smooth curve)

**Panel 3 (Bottom-Left): Feature Group Importance**
- Aggregates importance by feature type
- Shows which body part/parameter type matters most
- Groups like: body_pose, expression_component, shape_beta, etc.

**Panel 4 (Bottom-Right): Summary Statistics**
- Prediction label and confidence
- Number of frames and features analyzed
- Quick reference of top 5 features

### 2. `feature_ablation_report.txt`

**Detailed text report with:**
- Prediction confidence and probabilities
- Complete ranked list of all features by importance
- Impact classification: CRITICAL (>10%), HIGH (>5%), MEDIUM
- Interpretation guide

## Understanding the Results

### What Does Confidence Drop Mean?

```
Original prediction: 95% confidence (FAKE)
After removing feature X: 87% confidence (FAKE)
Confidence drop: 95% - 87% = 8%

→ Feature X was moderately important for this prediction
```

### Interpreting Feature Importance

| Conf Drop | Impact Level | Meaning |
|-----------|-------------|---------|
| > 10% | CRITICAL | Removing this feature drastically changes prediction |
| 5-10% | HIGH | Feature significantly contributes to prediction |
| 1-5% | MEDIUM | Feature somewhat contributes |
| < 1% | LOW | Feature barely affects prediction |

### What Each Feature Group Means

- **body_pose_[joint]_rot_x/y/z**: Rotation angles of body joints (e.g., shoulder, elbow)
- **expression_component_0-9**: Facial expression parameters (truncated from 50)
- **shape_beta_0-9**: Body shape/mesh parameters
- **left_hand_component_0-14**: Left hand pose (truncated from 45)
- **right_hand_component_0-14**: Right hand pose (truncated from 45)
- **global_orient_x/y/z**: Overall body orientation
- **camera_scale**: Camera scale parameter
- **camera_translation_x/y/z**: Camera position

## Examples

### Example 1: Fake Video with Motion Artifacts

**Top features:**
1. `body_pose_left_shoulder_rot_x` (8.5% drop)
2. `expression_component_3` (7.2% drop)
3. `body_pose_head_rot_y` (6.8% drop)

**Interpretation:** Model detected unnatural shoulder/head motion and facial expressions. These are common deepfake artifacts.

### Example 2: Real Video with Natural Motion

**Top features:**
1. `body_pose_pelvis_rot_x` (2.1% drop)
2. `camera_translation_z` (1.8% drop)
3. `body_pose_spine3_rot_z` (1.5% drop)

**Interpretation:** Many features contribute equally. No single feature is critical. This suggests balanced, natural motion pattern typical of real videos.

### Example 3: Low Confidence Prediction

**Top features:**
1. `expression_component_5` (0.8% drop)
2. `left_hand_component_7` (0.6% drop)
3. `shape_beta_2` (0.5% drop)

**Interpretation:** No feature is critically important. Model is uncertain. Manual inspection recommended.

## Combining with Other Analyses

### Ablation + Transformer Attention

1. Use **Transformer Attention** to find which **FRAMES** matter
2. Use **Feature Ablation** to find which **FEATURES** matter
3. Combined: "Frames 25-35 were critical, and within those, expression_component_3 was key"

### Ablation + Feature Importance (Dataset Level)

- **Dataset Feature Importance**: Which features are generally important across all videos
- **Single Video Ablation**: Which features mattered for THIS specific video
- Comparison reveals if prediction follows general patterns or uses specific anomalies

## Tips for Interpretation

### ✓ Do This
- Look at top 10-15 features, not just top 5
- Consider feature groups, not just individual features
- Compare confidence drops (relative importance matters more than absolute)
- Combine with visual inspection of flagged frames
- Cross-reference with transformer attention analysis

### ✗ Avoid This
- Don't over-interpret very small confidence drops (< 1%)
- Don't assume one feature explains everything
- Don't use ablation alone for low-confidence predictions
- Don't expect identical results across different model types (LSTM vs Transformer)

## Troubleshooting

### "feature_233" Instead of Feature Names

**Problem**: Output shows generic names instead of meaningful ones
**Solution**: Provide `--data-dir` argument so the tool can infer feature structure from data

### Very Slow Analysis

**Problem**: Feature ablation takes a long time
**Reason**: Need to run model inference 119+ times (once per feature)
**Solution**: Normal behavior. Patience is required, especially for Transformer models.

### Inconsistent Results

**Problem**: Running twice gives different results
**Reason**: Model has dropout layers, which are random during inference
**Solution**: Expected. Consider running multiple times and averaging results for critical decisions.

## Advanced: Understanding Ablation vs Other Methods

| Method | What It Shows | Pros | Cons |
|--------|---------------|------|------|
| **Feature Ablation** | Individual feature importance (one-at-a-time) | Simple, interpretable | Ignores feature interactions |
| **SHAP Values** | Feature importance with interactions considered | Theoretically sound | Complex, slow |
| **Attention** | Which frames the model looked at | Fast, transparent | Doesn't explain why |
| **Gradients** | Feature sensitivity (local) | Very fast | Only captures linear behavior |

**Recommendation**: Use ablation for straightforward interpretation, combine with attention for complete picture.
