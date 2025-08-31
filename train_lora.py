#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for Legal Contract Analyzer

This script fine-tunes a transformer model using LoRA on the CUAD dataset
for legal document analysis tasks.
"""

import argparse
import logging
from pathlib import Path
import torch

from src.models.lora_trainer import LoRATrainer
from src.config import CUAD_CONFIG, MODEL_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning for Legal Contract Analysis")
    
    parser.add_argument(
        "--model_name",
        type=str,
        default=MODEL_CONFIG["base_model"],
        help="Base model name for fine-tuning"
    )
    
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/cuad",
        help="Path to CUAD dataset"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/lora_legal",
        help="Output directory for fine-tuned model"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=MODEL_CONFIG["num_epochs"],
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=MODEL_CONFIG["batch_size"],
        help="Training batch size"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=MODEL_CONFIG["learning_rate"],
        help="Learning rate"
    )
    
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        default=True,
        help="Use GPU for training"
    )
    
    parser.add_argument(
        "--max_samples",
        type=int,
        default=CUAD_CONFIG["max_samples"],
        help="Maximum number of samples to use for training"
    )
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.use_gpu and not torch.cuda.is_available():
        logger.warning("GPU requested but not available. Falling back to CPU.")
        args.use_gpu = False
    
    logger.info("Starting LoRA fine-tuning for legal contract analysis")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Dataset: {args.dataset_path}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Device: {'CUDA' if args.use_gpu else 'CPU'}")
    
    try:
        # Initialize LoRA trainer
        trainer = LoRATrainer(base_model_name=args.model_name)
        
        # Load CUAD dataset
        logger.info("Loading CUAD dataset...")
        train_dataset, val_dataset, test_dataset = trainer.load_cuad_dataset(args.dataset_path)
        
        # Setup training
        logger.info("Setting up training...")
        trainer.setup_training(output_dir=args.output_dir)
        
        # Train the model
        logger.info("Starting training...")
        results = trainer.train()
        
        # Print results
        logger.info("Training completed successfully!")
        logger.info(f"Training loss: {results['train_loss']:.4f}")
        logger.info(f"Test loss: {results['test_loss']:.4f}")
        logger.info(f"Test accuracy: {results['test_accuracy']:.4f}")
        logger.info(f"Training time: {results['training_time']:.2f} seconds")
        logger.info(f"Model saved to: {results['model_path']}")
        
        # Save training results
        results_file = Path(args.output_dir) / "training_results.json"
        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Training results saved to: {results_file}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main() 