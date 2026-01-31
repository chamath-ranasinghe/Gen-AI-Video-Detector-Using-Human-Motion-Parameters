#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Video Frame Processor for SMPL-X Parameter Extraction
Processes videos, extracts frames, and generates body model parameters
"""

import sys
import os
import os.path as osp
from typing import Optional, List
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from loguru import logger
import torch
import torch.utils.data as dutils
from collections import defaultdict
import shutil

# Import necessary modules from the original code
from expose.data.datasets import ImageFolderWithBoxes
from expose.data.transforms import build_transforms
from expose.models.smplx_net import SMPLXNet
from expose.config import cfg
from expose.config.cmd_parser import set_face_contour
from expose.utils.checkpointer import Checkpointer
from expose.data.build import collate_batch
from expose.data.targets.image_list import to_image_list
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.transforms import Compose, ToTensor
import functools
from threadpoolctl import threadpool_limits


def extract_frames(video_path: str, output_folder: str, 
                   frame_rate: int = 1, max_frames: Optional[int] = None) -> List[str]:
    """
    Extract frames from video at specified frame rate
    
    Args:
        video_path: Path to input video
        output_folder: Folder to save extracted frames
        frame_rate: Extract every Nth frame (1 = every frame, 2 = every other frame, etc.)
        max_frames: Maximum number of frames to extract (None = all frames)
    
    Returns:
        List of paths to extracted frame images
    """
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        sys.exit(1)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    logger.info(f"Video: {video_path}")
    logger.info(f"Total frames: {total_frames}, FPS: {fps}")
    logger.info(f"Extracting every {frame_rate} frame(s)")
    
    frame_paths = []
    frame_count = 0
    extracted_count = 0
    
    pbar = tqdm(total=total_frames, desc="Extracting frames")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_rate == 0:
            frame_filename = f"frame_{frame_count:06d}.jpg"
            frame_path = osp.join(output_folder, frame_filename)
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            extracted_count += 1
            
            if max_frames is not None and extracted_count >= max_frames:
                break
        
        frame_count += 1
        pbar.update(1)
    
    cap.release()
    pbar.close()
    
    logger.info(f"Extracted {extracted_count} frames from {frame_count} total frames")
    return frame_paths


def detect_bodies_in_frames(frame_paths: List[str], 
                            device: torch.device,
                            batch_size: int = 4,
                            min_score: float = 0.5) -> tuple:
    """
    Detect human bodies in frames using Keypoint R-CNN
    
    Returns:
        Tuple of (filtered_frame_paths, bounding_boxes)
    """
    rcnn_model = keypointrcnn_resnet50_fpn(pretrained=True)
    rcnn_model.eval()
    rcnn_model = rcnn_model.to(device=device)
    
    transform = Compose([ToTensor()])
    
    img_paths = []
    bboxes = []
    
    # Process frames in batches
    for i in tqdm(range(0, len(frame_paths), batch_size), desc="Detecting bodies"):
        batch_paths = frame_paths[i:i + batch_size]
        batch_imgs = []
        
        for path in batch_paths:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = transform(img)
            batch_imgs.append(img_tensor.to(device=device))
        
        with torch.no_grad():
            outputs = rcnn_model(batch_imgs)
        
        for idx, output in enumerate(outputs):
            if len(output['boxes']) == 0:
                continue
            
            # Get the highest scoring detection
            best_idx = torch.argmax(output['scores']).item()
            if output['scores'][best_idx].item() >= min_score:
                bbox = output['boxes'][best_idx].detach().cpu().numpy()
                img_paths.append(batch_paths[idx])
                bboxes.append(bbox)
    
    logger.info(f"Detected bodies in {len(img_paths)} out of {len(frame_paths)} frames")
    return img_paths, bboxes


def process_frames_with_smplx(img_paths: List[str],
                              bboxes: List[np.ndarray],
                              exp_cfg,
                              device: torch.device,
                              scale_factor: float = 1.2,
                              num_workers: int = 0) -> dutils.DataLoader:
    """
    Create dataloader for SMPL-X processing
    """
    dataset_cfg = exp_cfg.get('datasets', {})
    body_dsets_cfg = dataset_cfg.get('body', {})
    body_transfs_cfg = body_dsets_cfg.get('transforms', {})
    transforms = build_transforms(body_transfs_cfg, is_train=False)
    batch_size = body_dsets_cfg.get('batch_size', 8)
    
    expose_dset = ImageFolderWithBoxes(
        img_paths, bboxes, 
        scale_factor=scale_factor,
        transforms=transforms
    )
    
    expose_collate = functools.partial(
        collate_batch, 
        use_shared_memory=num_workers > 0,
        return_full_imgs=True
    )
    
    expose_dloader = dutils.DataLoader(
        expose_dset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=expose_collate,
        drop_last=False,
        pin_memory=True,
    )
    
    return expose_dloader


@torch.no_grad()
def extract_smplx_parameters(dloader: dutils.DataLoader,
                            model: SMPLXNet,
                            device: torch.device,
                            output_folder: str) -> List[dict]:
    """
    Extract SMPL-X parameters from frames
    
    Returns:
        List of parameter dictionaries for each frame
    """
    model = model.eval()
    all_params = []
    
    for bidx, batch in enumerate(tqdm(dloader, desc="Extracting SMPL-X params")):
        full_imgs_list, body_imgs, body_targets = batch
        
        if full_imgs_list is None:
            continue
        
        full_imgs = to_image_list(full_imgs_list)
        body_imgs = body_imgs.to(device=device)
        body_targets = [target.to(device) for target in body_targets]
        full_imgs = full_imgs.to(device=device)
        
        model_output = model(body_imgs, body_targets, 
                           full_imgs=full_imgs, device=device)
        
        body_output = model_output.get('body', {})
        num_stages = body_output.get('num_stages', 3)
        stage_n_out = body_output.get(f'stage_{num_stages - 1:02d}', {})
        
        camera_parameters = body_output.get('camera_parameters', {})
        camera_scale = camera_parameters['scale'].detach().cpu().numpy()
        camera_transl = camera_parameters['translation'].detach().cpu().numpy()
        
        # Extract parameters for each frame in batch
        for idx in range(len(body_targets)):
            fname = body_targets[idx].get_field('fname')
            frame_params = {
                'frame_name': fname,
                'camera_scale': camera_scale[idx],
                'camera_translation': camera_transl[idx],
            }
            
            # Extract body parameters
            for key, val in stage_n_out.items():
                if torch.is_tensor(val):
                    frame_params[key] = val[idx].detach().cpu().numpy()
                else:
                    frame_params[key] = val
            
            all_params.append(frame_params)
    
    # Save all parameters
    params_file = osp.join(output_folder, 'all_frame_params.npz')
    np.savez_compressed(params_file, params=all_params)
    logger.info(f"Saved parameters for {len(all_params)} frames to {params_file}")
    
    return all_params


def main():
    parser = argparse.ArgumentParser(
        description='Extract SMPL-X parameters from video frames'
    )
    parser.add_argument('--frame-rate', type=int, default=5,
                       help='Extract every Nth frame (1=all frames, 5=every 5th frame)')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum number of frames to process')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for processing')
    parser.add_argument('--min-score', type=float, default=0.5,
                       help='Minimum detection score for body detection')
    parser.add_argument('--keep-frames', action='store_true',
                       help='Keep extracted frame images (default: delete after processing)')
    parser.add_argument('--exp-opts', default=[], nargs='*',
                       help='Extra configuration options')
    
    args = parser.parse_args()
    
    # Hardcoded path for EXPOSE config
    EXPOSE_CONFIG = './data/conf.yaml'
    
    # Get script directory and setup paths
    script_dir = osp.dirname(osp.abspath(__file__))
    DATASET_FOLDER = osp.join(script_dir, 'test_dataset')
    OUTPUT_BASE_FOLDER = './processed_test_videos'  # Base folder for all processed videos
    
    # Check if dataset folder exists
    if not osp.exists(DATASET_FOLDER):
        logger.error(f"Dataset folder not found: {DATASET_FOLDER}")
        logger.error("Please create a 'dataset' folder with 'real' and 'fake' subfolders")
        sys.exit(1)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        logger.warning('CUDA not available, using CPU (will be slow)')
    
    # Create output directory structure
    output_base = osp.expanduser(osp.expandvars(OUTPUT_BASE_FOLDER))
    os.makedirs(output_base, exist_ok=True)
    
    # Process both 'real' and 'fake' folders
    video_classes = ['real', 'fake']
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    
    model = None  # Will be initialized on first use
    
    for video_class in video_classes:
        video_folder = osp.join(DATASET_FOLDER, video_class)
        
        # Check if class folder exists
        if not osp.exists(video_folder):
            logger.warning(f"Folder not found: {video_folder}, skipping '{video_class}' class")
            continue
        
        # Create output class folder
        class_folder = osp.join(output_base, video_class)
        os.makedirs(class_folder, exist_ok=True)
        
        # Get list of video files
        video_files = [f for f in os.listdir(video_folder) 
                       if osp.isfile(osp.join(video_folder, f)) and 
                       osp.splitext(f)[1].lower() in video_extensions]
        
        if len(video_files) == 0:
            logger.warning(f"No video files found in {video_folder}")
            continue
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing {len(video_files)} videos from '{video_class}' class")
        logger.info(f"Input folder: {video_folder}")
        logger.info(f"Output folder: {class_folder}")
        logger.info(f"{'='*70}\n")
        
        # Process each video in this class
        for video_idx, video_file in enumerate(video_files):
            video_path = osp.join(video_folder, video_file)
            video_name = osp.splitext(video_file)[0]
            
            logger.info(f"\n{'-'*60}")
            logger.info(f"[{video_class.upper()}] Processing video {video_idx + 1}/{len(video_files)}: {video_file}")
            logger.info(f"{'-'*60}")
            
            # Create output folder for this video
            video_output_folder = osp.join(class_folder, video_name)
            os.makedirs(video_output_folder, exist_ok=True)
            
            frames_folder = osp.join(video_output_folder, 'frames')
            params_folder = video_output_folder  # Save params directly in video folder
            
            # Step 1: Extract frames
            logger.info("Step 1: Extracting frames from video")
            try:
                frame_paths = extract_frames(
                    video_path,
                    frames_folder,
                    frame_rate=args.frame_rate,
                    max_frames=args.max_frames
                )
            except Exception as e:
                logger.error(f"Failed to extract frames: {e}")
                continue
            
            if len(frame_paths) == 0:
                logger.error("No frames extracted! Skipping this video.")
                continue
            
            # Step 2: Detect bodies in frames
            logger.info("Step 2: Detecting human bodies in frames")
            try:
                img_paths, bboxes = detect_bodies_in_frames(
                    frame_paths,
                    device,
                    batch_size=args.batch_size,
                    min_score=args.min_score
                )
            except Exception as e:
                logger.error(f"Failed to detect bodies: {e}")
                if not args.keep_frames:
                    shutil.rmtree(frames_folder)
                continue
            
            if len(img_paths) == 0:
                logger.warning("No bodies detected in any frames! Skipping this video.")
                if not args.keep_frames:
                    shutil.rmtree(frames_folder)
                continue
            
            # Step 3: Setup SMPL-X model (only once)
            if model is None:
                logger.info("Step 3: Loading SMPL-X model (first time only)")
                cfg.merge_from_file(EXPOSE_CONFIG)
                cfg.merge_from_list(args.exp_opts)
                cfg.datasets.body.batch_size = args.batch_size
                cfg.is_training = False
                
                use_face_contour = cfg.datasets.use_face_contour
                set_face_contour(cfg, use_face_contour=use_face_contour)
                
                model = SMPLXNet(cfg)
                model = model.to(device=device)
                
                output_folder_cfg = cfg.output_folder
                checkpoint_folder = osp.join(output_folder_cfg, cfg.checkpoint_folder)
                checkpointer = Checkpointer(model, save_dir=checkpoint_folder, 
                                           pretrained=cfg.pretrained)
                checkpointer.load_checkpoint()
                logger.info("SMPL-X model loaded successfully")
            
            # Step 4: Create dataloader
            logger.info("Step 4: Preparing data for SMPL-X processing")
            try:
                dloader = process_frames_with_smplx(img_paths, bboxes, cfg, device)
            except Exception as e:
                logger.error(f"Failed to create dataloader: {e}")
                if not args.keep_frames:
                    shutil.rmtree(frames_folder)
                continue
            
            # Step 5: Extract parameters
            logger.info("Step 5: Extracting SMPL-X parameters")
            try:
                with threadpool_limits(limits=1):
                    all_params = extract_smplx_parameters(dloader, model, device, params_folder)
            except Exception as e:
                logger.error(f"Failed to extract parameters: {e}")
                if not args.keep_frames:
                    shutil.rmtree(frames_folder)
                continue
            
            # Cleanup frames if requested
            if not args.keep_frames:
                logger.info("Cleaning up frame images")
                shutil.rmtree(frames_folder)
            
            logger.info(f"✓ Successfully processed {len(all_params)} frames from {video_file}")
            logger.info(f"✓ Output saved to: {video_output_folder}")
    
    logger.info(f"\n{'='*70}")
    logger.info(f"ALL PROCESSING COMPLETE!")
    logger.info(f"Results saved to: {output_base}")
    logger.info(f"{'='*70}")


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    main()