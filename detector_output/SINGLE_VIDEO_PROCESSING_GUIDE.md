# Single Video Processing - Usage Guide

## Overview

The `video_processor.py` now supports processing a **single video file** without needing to organize it into the batch directory structure.

## When to Use Single Video Mode

- Processing one video at a time for testing/debugging
- Quick parameter extraction for manual inspection
- Ablation analysis on a specific video (paired with explainability toolkit)
- Processing videos outside the main dataset

## Command Usage

### Basic: Process single video with auto-generated output

```bash
python video_processor.py \
  --video /path/to/video.mp4 \
  --video-label fake
```

This will:
1. Extract the video filename
2. Create output folder: `./processed_single_videos/fake/video_name/`
3. Extract frames, detect bodies, and extract SMPL-X parameters
4. Delete temporary frame images (keep with `--keep-frames`)

### With custom output folder

```bash
python video_processor.py \
  --video /path/to/video.mp4 \
  --video-label real \
  --output ./my_custom_output/
```

### With additional options

```bash
python video_processor.py \
  --video ./test_videos/sample.mp4 \
  --video-label fake \
  --output ./extracted_params/ \
  --frame-rate 1 \
  --max-frames 100 \
  --batch-size 16 \
  --keep-frames
```

## Arguments

### Video Selection
- `--video`: (Required for single mode) Path to the video file
- `--video-label`: Label for the video - `real` or `fake` (default: `fake`)
- `--output`: Custom output folder path (optional)

### Processing
- `--frame-rate`: Extract every Nth frame (default: 5, meaning every 5th frame)
  - `1` = extract every frame (slow, detailed)
  - `5` = extract every 5th frame (faster, moderate detail)
  - `10` = extract every 10th frame (fastest, sparse)
- `--max-frames`: Maximum frames to process (optional, default: None = all)
- `--batch-size`: Batch size for GPU processing (default: 8)
- `--min-score`: Minimum detection score for body detection (default: 0.5)
- `--keep-frames`: Keep extracted frame images instead of deleting them
- `--exp-opts`: Extra EXPOSE configuration options (advanced)

## Output Structure

When processing a single video, the output structure is:

```
processed_single_videos/
└── [video_label]/
    └── [video_name]/
        ├── frames/
        │   ├── frame_000000.jpg
        │   ├── frame_000005.jpg
        │   └── ...
        └── params/
            └── all_frame_params.npz
```

If `--keep-frames` is NOT used, the `frames/` folder is deleted after processing.

## Examples

### Example 1: Quick test with a fake video

```bash
python video_processor.py \
  --video ./test_videos/deepfake_sample.mp4 \
  --video-label fake
```

Output: `./processed_single_videos/fake/deepfake_sample/`

### Example 2: Real video with frame preservation

```bash
python video_processor.py \
  --video ./videos/real_person.mp4 \
  --video-label real \
  --frame-rate 1 \
  --keep-frames
```

Output: `./processed_single_videos/real/real_person/`
- Extracts every frame (frame-rate=1)
- Keeps frame images for manual inspection

### Example 3: Fast processing with custom output

```bash
python video_processor.py \
  --video ./footage/unknown_video.mp4 \
  --video-label fake \
  --frame-rate 10 \
  --max-frames 50 \
  --output ./quick_analysis/
```

Output: `./quick_analysis/`
- Extracts every 10th frame (fast)
- Limits to 50 frames maximum
- Useful for quick analysis

### Example 4: For ablation analysis

```bash
# First extract parameters from a single video
python video_processor.py \
  --video ./my_video.mp4 \
  --video-label fake \
  --output ./temp_params/

# Then run ablation analysis
python explainability_toolkit.py \
  --model-path ./detector_output/best_model.pth \
  --model-type transformer \
  --task ablation \
  --video-params ./temp_params/params/all_frame_params.npz \
  --output-dir ./ablation_results
```

## Comparing Single Video vs Batch Mode

| Aspect | Single Video Mode | Batch Mode |
|--------|------------------|-----------|
| **Activation** | `--video <path>` argument | No argument (default) |
| **Input** | Single video file | Directory structure |
| **Output Location** | `./processed_single_videos/` | `./processed_test_videos/` |
| **Use Case** | Testing, individual analysis | Dataset processing |
| **Speed** | Faster (one video) | Slower (many videos) |
| **Setup** | Minimal | Requires directory structure |

## Tips

✓ **Use small frame-rate (e.g., 10) for quick tests**
```bash
--frame-rate 10 --max-frames 30
```

✓ **Use frame-rate 1 for thorough analysis**
```bash
--frame-rate 1
```

✓ **Keep frames for manual verification**
```bash
--keep-frames
```

✓ **Process high-quality videos with higher batch size**
```bash
--batch-size 16
```

## Troubleshooting

### "Video file not found"
Check that the path is correct and the file exists:
```bash
ls -la /path/to/video.mp4
```

### "No bodies detected in any frames"
This usually means:
- Video quality is too low
- Increase `--min-score` (e.g., `0.3`)
- Try increasing `--frame-rate` (extract more frames)
- Video may not contain clear body shapes

### Out of memory errors
Reduce batch size:
```bash
--batch-size 4
```

### Processing is very slow
Use fewer frames:
```bash
--frame-rate 5 --max-frames 100
```

## Integration with Explainability Toolkit

After extracting parameters from a single video, you can immediately use them with ablation analysis:

```bash
python explainability_toolkit.py \
  --model-path ./detector_output/best_model.pth \
  --model-type transformer \
  --task ablation \
  --video-params ./processed_single_videos/fake/my_video/params/all_frame_params.npz \
  --output-dir ./ablation/my_video
```

This workflow is perfect for:
1. **Detailed inspection**: Extract parameters from one video
2. **Model explanation**: Run ablation to see which features caused the prediction
3. **Manual verification**: Combine with frame images (keep with `--keep-frames`)
