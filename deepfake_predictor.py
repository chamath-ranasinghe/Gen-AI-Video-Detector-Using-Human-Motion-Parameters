#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Deepfake Video Predictor
Uses trained model to predict if videos are real or AI-generated
Supports both pre-extracted parameters and direct video input
"""

import os
import os.path as osp
import sys
import numpy as np
import argparse
from typing import Optional, Dict, List
import json
import cv2
from tqdm import tqdm
import shutil
import tempfile
import functools

import torch
import torch.nn as nn
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.transforms import Compose, ToTensor
import torch.nn.functional as F
import torch.utils.data as dutils
from loguru import logger

# Import from the training script
from deepfake_detector import DeepfakeDetectorLSTM, DeepfakeDetectorTransformer

# Import SMPL-X processing components
try:
    from expose.data.datasets import ImageFolderWithBoxes
    from expose.data.transforms import build_transforms
    from expose.models.smplx_net import SMPLXNet
    from expose.config import cfg
    from expose.config.cmd_parser import set_face_contour
    from expose.utils.checkpointer import Checkpointer
    from expose.data.build import collate_batch
    from expose.data.targets.image_list import to_image_list
    from threadpoolctl import threadpool_limits
    SMPLX_AVAILABLE = True
except ImportError:
    SMPLX_AVAILABLE = False
    logger.warning("SMPL-X modules not available. Direct video processing will be disabled.")


class VideoPredictor:
    """Predicts if a video is real or AI-generated"""
    
    def __init__(self, model_path: str, model_type: str = 'lstm',
                 expose_config: Optional[str] = None,
                 device: Optional[torch.device] = None):
        """
        Args:
            model_path: Path to trained model checkpoint (.pth file)
            model_type: 'lstm' or 'transformer'
            expose_config: Path to ExPose config (required for direct video processing)
            device: Computing device
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.expose_config = expose_config
        self.smplx_model = None
        
        # Load checkpoint
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get model configuration from checkpoint if available
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
            input_dim = config['input_dim']
            hidden_dim = config.get('hidden_dim', 256)
            num_layers = config.get('num_layers', 2)
            dropout = config.get('dropout', 0.3)
        else:
            # Default values
            print("Warning: Model config not found in checkpoint, using defaults")
            input_dim = 120
            hidden_dim = 256
            num_layers = 2
            dropout = 0.3
        
        # Initialize model
        if model_type == 'lstm':
            self.model = DeepfakeDetectorLSTM(input_dim, hidden_dim, num_layers, dropout)
        else:
            self.model = DeepfakeDetectorTransformer(input_dim, hidden_dim, 
                                                     num_layers=num_layers, dropout=dropout)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.input_dim = input_dim
        print(f"Model loaded successfully on {self.device}")
        print(f"Input dimension: {input_dim}")
    
    def _initialize_smplx_model(self):
        """Initialize SMPL-X model for video processing (lazy loading)"""
        if self.smplx_model is not None:
            return
        
        if not SMPLX_AVAILABLE:
            raise ImportError("SMPL-X modules not available. Please install ExPose.")
        
        if self.expose_config is None:
            raise ValueError("ExPose config path required for direct video processing")
        
        logger.info("Loading SMPL-X model for video processing...")
        cfg.merge_from_file(self.expose_config)
        cfg.datasets.body.batch_size = 8
        cfg.is_training = False
        
        use_face_contour = cfg.datasets.use_face_contour
        set_face_contour(cfg, use_face_contour=use_face_contour)
        
        self.smplx_model = SMPLXNet(cfg)
        self.smplx_model = self.smplx_model.to(device=self.device)
        
        output_folder_cfg = cfg.output_folder
        checkpoint_folder = osp.join(output_folder_cfg, cfg.checkpoint_folder)
        checkpointer = Checkpointer(self.smplx_model, save_dir=checkpoint_folder, 
                                   pretrained=cfg.pretrained)
        checkpointer.load_checkpoint()
        self.smplx_model.eval()
        
        self.smplx_cfg = cfg
        logger.info("SMPL-X model loaded successfully")
    
    def extract_features_from_frame_params(self, frame_param: dict) -> np.ndarray:
        """Extract features from a single frame's parameters"""
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
        
        # Body pose parameters
        if 'body_pose' in frame_param:
            add_feature(frame_param['body_pose'])
        
        # Global orientation
        if 'global_orient' in frame_param:
            add_feature(frame_param['global_orient'])
        
        # Shape parameters
        if 'betas' in frame_param:
            add_feature(frame_param['betas'])
        
        # Hand pose parameters (limit to 15 each)
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
    
    def _extract_frames(self, video_path: str, output_folder: str, 
                       frame_rate: int = 5) -> List[str]:
        """Extract frames from video"""
        os.makedirs(output_folder, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        
        frame_paths = []
        frame_count = 0
        extracted_count = 0
        
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
            
            frame_count += 1
        
        cap.release()
        logger.info(f"Extracted {extracted_count} frames from {frame_count} total frames")
        return frame_paths
    
    def _detect_bodies(self, frame_paths: List[str], batch_size: int = 4,
                      min_score: float = 0.5) -> tuple:
        """Detect human bodies in frames"""
        rcnn_model = keypointrcnn_resnet50_fpn(pretrained=True)
        rcnn_model.eval()
        rcnn_model = rcnn_model.to(device=self.device)
        
        transform = Compose([ToTensor()])
        
        img_paths = []
        bboxes = []
        
        for i in tqdm(range(0, len(frame_paths), batch_size), desc="Detecting bodies"):
            batch_paths = frame_paths[i:i + batch_size]
            batch_imgs = []
            
            for path in batch_paths:
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_tensor = transform(img)
                batch_imgs.append(img_tensor.to(device=self.device))
            
            with torch.no_grad():
                outputs = rcnn_model(batch_imgs)
            
            for idx, output in enumerate(outputs):
                if len(output['boxes']) == 0:
                    continue
                
                best_idx = torch.argmax(output['scores']).item()
                if output['scores'][best_idx].item() >= min_score:
                    bbox = output['boxes'][best_idx].detach().cpu().numpy()
                    img_paths.append(batch_paths[idx])
                    bboxes.append(bbox)
        
        logger.info(f"Detected bodies in {len(img_paths)} out of {len(frame_paths)} frames")
        return img_paths, bboxes
    
    def _extract_smplx_params(self, img_paths: List[str], bboxes: List[np.ndarray]) -> List[dict]:
        """Extract SMPL-X parameters from detected bodies"""
        dataset_cfg = self.smplx_cfg.get('datasets', {})
        body_dsets_cfg = dataset_cfg.get('body', {})
        body_transfs_cfg = body_dsets_cfg.get('transforms', {})
        transforms = build_transforms(body_transfs_cfg, is_train=False)
        batch_size = body_dsets_cfg.get('batch_size', 8)
        
        expose_dset = ImageFolderWithBoxes(img_paths, bboxes, 
                                          scale_factor=1.2, transforms=transforms)
        
        expose_collate = functools.partial(collate_batch, use_shared_memory=False,
                                          return_full_imgs=True)
        
        expose_dloader = dutils.DataLoader(expose_dset, batch_size=batch_size,
                                          num_workers=0, collate_fn=expose_collate,
                                          drop_last=False, pin_memory=True)
        
        all_params = []
        
        for batch in tqdm(expose_dloader, desc="Extracting SMPL-X params"):
            full_imgs_list, body_imgs, body_targets = batch
            
            if full_imgs_list is None:
                continue
            
            full_imgs = to_image_list(full_imgs_list)
            body_imgs = body_imgs.to(device=self.device)
            body_targets = [target.to(self.device) for target in body_targets]
            full_imgs = full_imgs.to(device=self.device)
            
            with torch.no_grad():
                model_output = self.smplx_model(body_imgs, body_targets, 
                                               full_imgs=full_imgs, device=self.device)
            
            body_output = model_output.get('body', {})
            num_stages = body_output.get('num_stages', 3)
            stage_n_out = body_output.get(f'stage_{num_stages - 1:02d}', {})
            
            camera_parameters = body_output.get('camera_parameters', {})
            camera_scale = camera_parameters['scale'].detach().cpu().numpy()
            camera_transl = camera_parameters['translation'].detach().cpu().numpy()
            
            for idx in range(len(body_targets)):
                frame_params = {
                    'camera_scale': camera_scale[idx],
                    'camera_translation': camera_transl[idx],
                }
                
                for key, val in stage_n_out.items():
                    if torch.is_tensor(val):
                        frame_params[key] = val[idx].detach().cpu().numpy()
                
                all_params.append(frame_params)
        
        return all_params
    
    def predict_from_params(self, params_path: str, max_frames: int = 100) -> Dict:
        """
        Predict from pre-extracted parameters
        
        Args:
            params_path: Path to .npz file containing frame parameters
            max_frames: Maximum number of frames to use
        
        Returns:
            Dictionary with prediction results
        """
        # Load parameters
        data = np.load(params_path, allow_pickle=True)
        params = data['params']
        
        # Extract features from each frame
        frame_features = []
        for frame_param in params[:max_frames]:
            features = self.extract_features_from_frame_params(frame_param)
            frame_features.append(features)
        
        num_actual_frames = len(frame_features)
        
        # Pad or truncate to max_frames
        if len(frame_features) < max_frames:
            padding = [np.zeros_like(frame_features[0])] * (max_frames - len(frame_features))
            frame_features.extend(padding)
        
        frame_features = np.stack(frame_features[:max_frames])
        
        # Convert to tensor and add batch dimension
        features_tensor = torch.FloatTensor(frame_features).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(features_tensor)
            probs = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probs[0, predicted_class].item()
        
        # Get probabilities for both classes
        prob_real = probs[0, 0].item()
        prob_fake = probs[0, 1].item()
        
        return {
            'prediction': 'AI-Generated' if predicted_class == 1 else 'Real',
            'predicted_class': predicted_class,
            'confidence': confidence * 100,
            'prob_real': prob_real * 100,
            'prob_fake': prob_fake * 100,
            'num_frames_analyzed': num_actual_frames
        }
    
    def predict_from_video(self, video_path: str, frame_rate: int = 5,
                          max_frames: int = 100, cleanup: bool = True) -> Dict:
        """
        Predict directly from video file
        
        Args:
            video_path: Path to video file
            frame_rate: Extract every Nth frame
            max_frames: Maximum number of frames to process
            cleanup: Whether to delete temporary files
        
        Returns:
            Dictionary with prediction results
        """
        if not SMPLX_AVAILABLE:
            raise ImportError("SMPL-X modules not available. Please install ExPose.")
        
        # Initialize SMPL-X model if needed
        self._initialize_smplx_model()
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix='video_prediction_')
        frames_dir = osp.join(temp_dir, 'frames')
        
        try:
            logger.info(f"Processing video: {video_path}")
            
            # Step 1: Extract frames
            logger.info("Step 1: Extracting frames...")
            frame_paths = self._extract_frames(video_path, frames_dir, frame_rate)
            
            if len(frame_paths) == 0:
                raise RuntimeError("No frames extracted from video")
            
            # Step 2: Detect bodies
            logger.info("Step 2: Detecting human bodies...")
            img_paths, bboxes = self._detect_bodies(frame_paths)
            
            if len(img_paths) == 0:
                raise RuntimeError("No human bodies detected in video")
            
            # Step 3: Extract SMPL-X parameters
            logger.info("Step 3: Extracting body parameters...")
            with threadpool_limits(limits=1):
                all_params = self._extract_smplx_params(img_paths, bboxes)
            
            if len(all_params) == 0:
                raise RuntimeError("Failed to extract body parameters")
            
            # Step 4: Extract features and predict
            logger.info("Step 4: Making prediction...")
            frame_features = []
            for frame_param in all_params[:max_frames]:
                features = self.extract_features_from_frame_params(frame_param)
                frame_features.append(features)
            
            num_actual_frames = len(frame_features)
            
            # Pad or truncate
            if len(frame_features) < max_frames:
                padding = [np.zeros_like(frame_features[0])] * (max_frames - len(frame_features))
                frame_features.extend(padding)
            
            frame_features = np.stack(frame_features[:max_frames])
            features_tensor = torch.FloatTensor(frame_features).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(features_tensor)
                probs = F.softmax(outputs, dim=1)
                predicted_class = torch.argmax(outputs, dim=1).item()
                confidence = probs[0, predicted_class].item()
            
            prob_real = probs[0, 0].item()
            prob_fake = probs[0, 1].item()
            
            return {
                'prediction': 'AI-Generated' if predicted_class == 1 else 'Real',
                'predicted_class': predicted_class,
                'confidence': confidence * 100,
                'prob_real': prob_real * 100,
                'prob_fake': prob_fake * 100,
                'num_frames_analyzed': num_actual_frames
            }
        
        finally:
            # Cleanup temporary files
            if cleanup and osp.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info("Cleaned up temporary files")


def print_prediction_result(result: Dict, video_name: str):
    """Pretty print prediction results"""
    print("\n" + "="*70)
    print(f"PREDICTION RESULTS FOR: {video_name}")
    print("="*70)
    print(f"\n{'Prediction:':<25} {result['prediction']}")
    print(f"{'Confidence:':<25} {result['confidence']:.2f}%")
    print(f"\n{'Probability (Real):':<25} {result['prob_real']:.2f}%")
    print(f"{'Probability (Fake):':<25} {result['prob_fake']:.2f}%")
    print(f"\n{'Frames Analyzed:':<25} {result['num_frames_analyzed']}")
    print("="*70)
    
    # Visual bar
    print("\nConfidence Visualization:")
    bar_length = 50
    real_bars = int(result['prob_real'] / 100 * bar_length)
    fake_bars = bar_length - real_bars
    print(f"Real [{'█' * real_bars}{'░' * fake_bars}] Fake")
    print(f"     {result['prob_real']:.1f}%{' ' * (bar_length - 10)}{result['prob_fake']:.1f}%")
    print()


def batch_predict(model_path: str, params_dir: str, model_type: str = 'lstm',
                 true_label: Optional[str] = None) -> List[Dict]:
    """
    Predict on multiple videos and calculate accuracy if labels are known
    """
    predictor = VideoPredictor(model_path, model_type)
    
    # Find all parameter files
    param_files = []
    for root, dirs, files in os.walk(params_dir):
        for file in files:
            if file == 'all_frame_params.npz':
                param_files.append(osp.join(root, file))
    
    if len(param_files) == 0:
        print(f"No parameter files found in {params_dir}")
        return []
    
    print(f"\nFound {len(param_files)} videos to analyze")
    if true_label:
        print(f"True label: {true_label.upper()}")
    print()
    
    # Predict on all files
    results = []
    correct_predictions = 0
    
    for param_file in tqdm(param_files, desc="Processing videos"):
        video_name = osp.basename(osp.dirname(param_file))
        
        try:
            result = predictor.predict_from_params(param_file)
            result['video_name'] = video_name
            result['param_file'] = param_file
            results.append(result)
            
            # Check accuracy if true label is provided
            if true_label:
                predicted_label = 'fake' if result['predicted_class'] == 1 else 'real'
                if predicted_label == true_label.lower():
                    correct_predictions += 1
        
        except Exception as e:
            print(f"\nError processing {video_name}: {e}")
            continue
    
    # Print summary
    print("\n" + "="*70)
    print("BATCH PREDICTION SUMMARY")
    print("="*70)
    print(f"\nTotal videos processed: {len(results)}")
    
    # Count predictions
    real_count = sum(1 for r in results if r['predicted_class'] == 0)
    fake_count = sum(1 for r in results if r['predicted_class'] == 1)
    
    print(f"Predicted as Real: {real_count} ({real_count/len(results)*100:.1f}%)")
    print(f"Predicted as Fake: {fake_count} ({fake_count/len(results)*100:.1f}%)")
    
    # Calculate and display accuracy if true label provided
    if true_label:
        accuracy = (correct_predictions / len(results)) * 100
        print(f"\n{'ACCURACY:':<25} {accuracy:.2f}%")
        print(f"{'Correct Predictions:':<25} {correct_predictions}/{len(results)}")
    
    # Average confidence
    avg_confidence = np.mean([r['confidence'] for r in results])
    print(f"\nAverage Confidence: {avg_confidence:.2f}%")
    
    print("="*70)
    
    # Detailed results
    print("\nDETAILED RESULTS:")
    print("-"*70)
    for result in results:
        status = ""
        if true_label:
            predicted_label = 'fake' if result['predicted_class'] == 1 else 'real'
            correct = "✓" if predicted_label == true_label.lower() else "✗"
            status = f" [{correct}]"
        
        print(f"{result['video_name']:<30} → {result['prediction']:<15} "
              f"({result['confidence']:.1f}%){status}")
    print("-"*70)
    
    return results


def batch_predict_videos(model_path: str, video_dir: str, model_type: str = 'lstm',
                        expose_config: str = None, frame_rate: int = 5,
                        max_frames: int = 100, true_label: Optional[str] = None,
                        cleanup: bool = True) -> List[Dict]:
    """
    Predict on multiple videos directly
    """
    predictor = VideoPredictor(model_path, model_type, expose_config=expose_config)
    
    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    video_files = [f for f in os.listdir(video_dir)
                   if osp.isfile(osp.join(video_dir, f)) and
                   osp.splitext(f)[1].lower() in video_extensions]
    
    if len(video_files) == 0:
        print(f"No video files found in {video_dir}")
        return []
    
    print(f"\nFound {len(video_files)} videos to analyze")
    if true_label:
        print(f"True label: {true_label.upper()}")
    print()
    
    # Predict on all videos
    results = []
    correct_predictions = 0
    
    for video_file in tqdm(video_files, desc="Processing videos"):
        video_path = osp.join(video_dir, video_file)
        
        try:
            result = predictor.predict_from_video(
                video_path, frame_rate, max_frames, cleanup
            )
            result['video_name'] = video_file
            result['video_path'] = video_path
            results.append(result)
            
            # Check accuracy if true label is provided
            if true_label:
                predicted_label = 'fake' if result['predicted_class'] == 1 else 'real'
                if predicted_label == true_label.lower():
                    correct_predictions += 1
        
        except Exception as e:
            print(f"\nError processing {video_file}: {e}")
            continue
    
    # Print summary
    print("\n" + "="*70)
    print("BATCH PREDICTION SUMMARY")
    print("="*70)
    print(f"\nTotal videos processed: {len(results)}")
    
    real_count = sum(1 for r in results if r['predicted_class'] == 0)
    fake_count = sum(1 for r in results if r['predicted_class'] == 1)
    
    print(f"Predicted as Real: {real_count} ({real_count/len(results)*100:.1f}%)")
    print(f"Predicted as Fake: {fake_count} ({fake_count/len(results)*100:.1f}%)")
    
    if true_label:
        accuracy = (correct_predictions / len(results)) * 100
        print(f"\n{'ACCURACY:':<25} {accuracy:.2f}%")
        print(f"{'Correct Predictions:':<25} {correct_predictions}/{len(results)}")
    
    avg_confidence = np.mean([r['confidence'] for r in results])
    print(f"\nAverage Confidence: {avg_confidence:.2f}%")
    print("="*70)
    
    # Detailed results
    print("\nDETAILED RESULTS:")
    print("-"*70)
    for result in results:
        status = ""
        if true_label:
            predicted_label = 'fake' if result['predicted_class'] == 1 else 'real'
            correct = "✓" if predicted_label == true_label.lower() else "✗"
            status = f" [{correct}]"
        
        print(f"{result['video_name']:<30} → {result['prediction']:<15} "
              f"({result['confidence']:.1f}%){status}")
    print("-"*70)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Predict if videos are real or AI-generated'
    )
    
    # Model parameters
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--model-type', type=str, default='lstm',
                       choices=['lstm', 'transformer'],
                       help='Model architecture type')
    
    # Input options (video or params)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--video-path', type=str,
                            help='Path to video file for direct prediction')
    input_group.add_argument('--params-path', type=str,
                            help='Path to parameters file (all_frame_params.npz)')
    input_group.add_argument('--video-dir', type=str,
                            help='Directory containing video files for batch processing')
    input_group.add_argument('--params-dir', type=str,
                            help='Directory containing parameter files for batch processing')
    
    # For direct video processing
    parser.add_argument('--expose-config', type=str,
                       help='Path to ExPose config file (required for --video-path/--video-dir)')
    parser.add_argument('--frame-rate', type=int, default=5,
                       help='Extract every Nth frame from video')
    
    # Batch mode options
    parser.add_argument('--true-label', type=str, choices=['real', 'fake'],
                       help='True label for accuracy calculation (batch mode only)')
    
    # Processing parameters
    parser.add_argument('--max-frames', type=int, default=100,
                       help='Maximum number of frames to analyze')
    parser.add_argument('--keep-temp', action='store_true',
                       help='Keep temporary files (for video input)')
    
    # Output
    parser.add_argument('--save-results', type=str,
                       help='Path to save prediction results as JSON')
    
    args = parser.parse_args()
    
    # Validate arguments
    if (args.video_path or args.video_dir) and not args.expose_config:
        parser.error("--expose-config is required when using --video-path or --video-dir")
    
    if not osp.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    # Determine actual mode based on inputs
    if args.video_path or args.params_path:
        actual_mode = 'single'
    elif args.video_dir or args.params_dir:
        actual_mode = 'batch'
    else:
        parser.error("Must specify one of: --video-path, --params-path, --video-dir, --params-dir")
    
    # Initialize predictor
    print(f"Initializing predictor with {args.model_type.upper()} model...")
    
    # Run prediction
    if actual_mode == 'single':
        # Single video/params prediction
        if args.video_path:
            # Direct video prediction
            if not osp.exists(args.video_path):
                print(f"Error: Video file not found: {args.video_path}")
                sys.exit(1)
            
            predictor = VideoPredictor(args.model_path, args.model_type, 
                                      expose_config=args.expose_config)
            
            video_name = osp.basename(args.video_path)
            print(f"\nAnalyzing video: {video_name}")
            print("This may take several minutes...")
            
            result = predictor.predict_from_video(
                args.video_path, 
                frame_rate=args.frame_rate,
                max_frames=args.max_frames,
                cleanup=not args.keep_temp
            )
            print_prediction_result(result, video_name)
            
            if args.save_results:
                result['video_name'] = video_name
                result['video_path'] = args.video_path
                with open(args.save_results, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\nResults saved to: {args.save_results}")
        
        else:
            # Params prediction
            if not osp.exists(args.params_path):
                print(f"Error: Parameters file not found: {args.params_path}")
                sys.exit(1)
            
            predictor = VideoPredictor(args.model_path, args.model_type)
            
            video_name = osp.basename(osp.dirname(args.params_path))
            print(f"\nAnalyzing video: {video_name}")
            
            result = predictor.predict_from_params(args.params_path, args.max_frames)
            print_prediction_result(result, video_name)
            
            if args.save_results:
                result['video_name'] = video_name
                result['params_path'] = args.params_path
                with open(args.save_results, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\nResults saved to: {args.save_results}")
    
    else:
        # Batch prediction
        if args.video_dir:
            # Batch video prediction
            if not osp.exists(args.video_dir):
                print(f"Error: Video directory not found: {args.video_dir}")
                sys.exit(1)
            
            results = batch_predict_videos(
                args.model_path, args.video_dir, args.model_type,
                args.expose_config, args.frame_rate, args.max_frames,
                args.true_label, not args.keep_temp
            )
        else:
            # Batch params prediction
            if not osp.exists(args.params_dir):
                print(f"Error: Parameters directory not found: {args.params_dir}")
                sys.exit(1)
            
            results = batch_predict(
                args.model_path, args.params_dir, args.model_type,
                args.true_label
            )
        
        # Save results if requested
        if args.save_results and results:
            with open(args.save_results, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.save_results}")


if __name__ == '__main__':
    main()