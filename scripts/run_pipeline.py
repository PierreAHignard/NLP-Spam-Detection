#!/usr/bin/env python3
"""
Simple Air Quality ML Pipeline with Inline MLflow Integration

This pipeline includes MLflow logging directly in the main workflow without
utility functions, making it easy for students to understand.
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pipeline import DataProcessor, FeatureEngineer, ModelTrainer
from pipeline.evaluator import Evaluator
from utils.config import MODEL_TYPES

from utils.logger import get_logger, set_log_level, log_level_from_string, LogLevel
from utils.utils import format_time_elapsed

# TODO Import parameter grids for optimization (Workshop 3)

# TODO Import MLflow (Workshop 4)


def run_pipeline(args):
    """
    Run the complete air quality prediction pipeline with inline MLflow integration.
    """
    start_time = time.time()
    logger = get_logger()

    # TODO Add MLflow setup and run start (Workshop 4)
        # Configuration MLflow simple
        
        # Create descriptive run name
        
        
        # Set tags for Dataset and Model columns in MLflow UI
            # Dataset tags (for Dataset column)
            
            # Model tags (for Model column)
            
            # Pipeline tags
        
        # Log pipeline configuration parameters
    
    try:
        # Pipeline header with configuration
        logger.header("AIR QUALITY ML PIPELINE")
        with logger.indent():
            logger.info(f"Model: {args.model}")
            logger.info(f"Features: {args.n_features}")
            logger.info(f"Selection method: {args.method}")
            logger.info(f"Optimization: {'Enabled' if args.optimize else 'Disabled'}")
            logger.info(f"MLflow tracking: {'Enabled' if args.mlflow else 'Disabled'}")
            if args.mlflow:
                logger.info("Final model will be retrained on all data and registered in MLflow")

        # Initialize components
        processor = DataProcessor()
        engineer = FeatureEngineer()
        trainer = ModelTrainer()

        # Step 1: Data Loading and Preprocessing
        logger.step("Data Loading and Preprocessing", 1)
        with logger.timer("Data loading and preprocessing"):
            train_data, test_data = processor.load_and_preprocess()

        # TODO Add MLflow dataset logging (Workshop 4)
            # Log dataset inline (no separate function)
            
            # Log dataset metrics

        # Step 2: Feature Engineering
        logger.step("Feature Engineering", 2)
        with logger.timer("Feature engineering"):
            train_features, test_features = engineer.extract_all_features(train_data, test_data)

        with logger.indent():
            logger.data_info(f"Original features: {train_data.shape[1]}")
            logger.feature_info(f"Features after engineering: {train_features.shape[1]}")

        # Step 3: Feature Selection
        logger.step("Feature Selection", 3)
        with logger.timer("Feature selection"):
            selected_features = engineer.select_best_features(
                train_features, 
                method=args.method, 
                n_features=args.n_features
            )

        logger.feature_info(f"Selected {len(selected_features)} features")

        # TODO Add MLflow feature selection logging (Workshop 4)
            # Note: engineer.select_best_features() already logs MLflow parameters
            # We just log additional pipeline-specific info with different parameter names

        # Step 4: Cross-Validation Evaluation
        logger.step("Cross-Validation Evaluation", 4)

        # Initialize evaluator
        evaluator = Evaluator()

        # TODO Create model for cross-validation

        # TODO Prepare data X, y, and groups for cross-validation

        if not args.optimize:
            with logger.timer("Cross-validation"):
                # TODO Standard cross-validation using Evaluator
        # TODO Add hyperparameter optimization logic (Workshop 3)

            # Get parameter grid for the model
            # If no grid is defined, use default parameters
            # If grid is defined, perform optimization        
                # Perform hyperparameter optimization

                    # Add MLflow hyperparameter optimization logging (Workshop 4)

                    # Use optimized model for final evaluation

                    # Quick evaluation to get full cv_results format

        # Extract results for compatibility
        mean_rmse = cv_results['rmse_mean']
        std_rmse = cv_results['rmse_std']
        mean_r2 = cv_results['r2_mean']
        std_r2 = cv_results['r2_std']

        # TODO Add MLflow model and results logging (Workshop 4)
            # Log cross-validation results
            
            # Log the trained model (for Model column)
                    # Step 1: Prepare clean data for MLflow (avoid warnings)
                    # Remove rows with missing values and convert to float64

                    # Step 2: Create MLflow model signature using clean data
                    # The signature describes input/output format for the model

                    # Step 3: Create descriptive model name (appears in Model column)

                    # Step 4: Prepare input example for MLflow documentation
                    # This shows users what kind of data the model expects
                    
                    # Log model to MLflow
                    
                    # Register model in MLflow Model Registry
                            
                

        # TODO Step 5: Save Model (Workshop 4)
            
            # Retrain final model on all training data (best practice)
            
            # Log the final model to MLflow
                # Prepare clean data for MLflow model signature
                
                # Create model name for MLflow
                
                # Log final model to MLflow
                
                # Register model in MLflow Model Registry
                
                
                

        # Prepare results for summary
        model_name = args.model
        cv_results_dict = {
            'rmse_mean': mean_rmse,
            'rmse_std': std_rmse,
            'r2_mean': mean_r2,
            'r2_std': std_r2
        }

        results = {
            'model_type': args.model,
            'cv_results': cv_results_dict,
            'selected_features': selected_features
        }

        # Step 6: Results Summary
        logger.step("Results Summary", 6)

        end_time = time.time()
        execution_time = format_time_elapsed(start_time, end_time)

        summary = {
            'Model': model_name,
            'RMSE': f"{cv_results_dict.get('rmse_mean', 'N/A'):.3f}",
            'R²': f"{cv_results_dict.get('r2_mean', 'N/A'):.3f}",
            'Features': len(selected_features),
            'Selection Method': args.method,
            'Optimized': args.optimize,
            'Execution Time': execution_time
        }

        logger.results_summary(summary)

        # TODO Add MLflow final results logging (Workshop 4)


        logger.pipeline_complete(end_time - start_time)

        return {
            'results': results,
            'summary': summary,
            'selected_features': selected_features,
            'execution_time': execution_time
        }
    # TODO End MLflow run (Workshop 4)


def parse_arguments():
    """Parse command line arguments."""
    
    parser = argparse.ArgumentParser(
        description="Run Air Quality ML Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--model', type=str, default='linear',
        choices=MODEL_TYPES,
        help='Model type to train'
    )
    
    parser.add_argument(
        '--n-features', type=int, default=15,
        help='Number of features to select'
    )
    
    parser.add_argument(
        '--method', type=str, default='selectkbest',
        choices=['selectkbest', 'rfe'],
        help='Feature selection method'
    )
    
    parser.add_argument(
        '--optimize', action='store_true',
        help='Enable hyperparameter optimization using GridSearchCV'
    )
    
    parser.add_argument(
        '--compare', action='store_true',
        help='Compare multiple models instead of training single model'
    )
     
    parser.add_argument(
        '--verbose', action='store_true',
        help='Enable verbose output (deprecated, use --log-level verbose)'
    )
    
    parser.add_argument(
        '--log-level', type=str, default='normal',
        choices=['silent', 'normal', 'verbose'],
        help='Logging level: silent (no output), normal (main steps), verbose (all details)'
    )

    # TODO Add MLflow tracking argument --mlflow (Workshop 4)
    
    return parser.parse_args()


def main():
    """Main entry point."""
    
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Configure logging level
        if args.verbose:
            # Support legacy --verbose flag
            log_level = LogLevel.VERBOSE
        else:
            log_level = log_level_from_string(args.log_level)
        
        set_log_level(log_level)
        
        # Run pipeline
        run_pipeline(args)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
