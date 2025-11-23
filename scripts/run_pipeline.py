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
from sklearn.metrics import r2_score

from utils.config import TARGET_COL, DATA_PATH, MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pipeline import DataProcessor, FeatureEngineer, ModelTrainer
from pipeline.evaluator import Evaluator
from utils.config import MODEL_TYPES

from utils.logger import get_logger, set_log_level, log_level_from_string, LogLevel
from utils.utils import format_time_elapsed

# Import parameter grids for optimization (Workshop 3)

from utils.config import DEFAULT_PARAM_GRIDS

# Import MLflow (Workshop 4)
import mlflow


def run_pipeline(args):
    """
    Run the complete air quality prediction pipeline with inline MLflow integration.
    """
    start_time = time.time()
    logger = get_logger()

    # Add MLflow setup and run start (Workshop 4)
        # Configuration MLflow simple
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        # Create descriptive run name
    mlflow.start_run(run_name="run")

        # Set tags for Dataset and Model columns in MLflow UI
    # PARAMS
    mlflow.log_param("model.type", args.model)
    mlflow.log_param("model.optimize", args.optimize)
    mlflow.log_param("features.n_features", args.n_features)
    mlflow.log_param("features.selection_method", args.method)

    # TAGS
    mlflow.set_tag("dataset.path", DATA_PATH)
    mlflow.set_tag("pipeline.mlflow_enabled", args.mlflow)
    mlflow.set_tag("mlflow.note.content",
                   f"Pipeline with {args.model} model and {args.n_features} features")


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
            
        # Add MLflow dataset logging (Workshop 4)
        if mlflow.active_run():
            # Log dataset inline (no separate function)
            # Log dataset metrics
            mlflow.log_param("dataset.train_size", len(train_data))
            mlflow.log_param("dataset.test_size", len(test_data))
            mlflow.log_metric("dataset.train_rows", len(train_data))
            mlflow.log_metric("dataset.test_rows", len(test_data))
            mlflow.log_param("dataset.initial_features", train_data.shape[1])

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

        # Add MLflow feature selection logging (Workshop 4)
            # Note: engineer.select_best_features() already logs MLflow parameters
            # We just log additional pipeline-specific info with different parameter names

        if mlflow.active_run():
            mlflow.log_metric("features.n_engineered", train_features.shape[1])

            mlflow.log_dict(
                {"selected_features": selected_features},
                "selected_features.json"
            )

        # Step 4: Cross-Validation Evaluation
        logger.step("Cross-Validation Evaluation", 4)

        # Initialize evaluator
        evaluator = Evaluator()

        # Create model for cross-validation
        if args.model in MODEL_TYPES:
            model = trainer.create_model(args.model)
        else:
            raise Exception(f"The input model '{args.model}' is not recognised")

        # Prepare data X, y, and groups for cross-validation

        X = train_features[selected_features]
        y = train_features[TARGET_COL]
        groups = train_data["city"]

        if not args.optimize:
            # CASE 1 — no optimization
            with logger.timer("Cross-validation"):
                cv_results = evaluator.cross_validate_model(model, X, y, groups)

        else:
            # CASE 2 — optimization requested

            # Check if the model has a grid
            param_grid = DEFAULT_PARAM_GRIDS.get(args.model, None)

            # If no grid, fallback to simple CV
            if param_grid is None:
                logger.warning(f"No parameter grid for '{args.model}'. Running standard CV instead.")

                with logger.timer("Cross-validation"):
                    cv_results = evaluator.cross_validate_model(model, X, y, groups)

            else:
                # Grid exists → perform optimization
                with logger.timer("Hyperparameter optimization"):
                    best_model, best_params, best_score = evaluator.hyperparameter_optimization_cv(model, param_grid, X, y, groups)

                #calculate r2
                y_pred = best_model.predict(X)
                r2_best = r2_score(y, y_pred)
                # Build your simple result dict 
                cv_results = {
                    "rmse_mean": best_score,
                    "rmse_std": 0,
                    "r2_mean": r2_best,
                    "r2_std": 0
                }   

                # Add MLflow hyperparameter optimization logging (Workshop 4)
                    # Use optimized model for final evaluation
                    # Quick evaluation to get full cv_results format
                if mlflow.active_run():
                    # Log hyperparameter optimization details
                    mlflow.log_param("optimization.method", "GridSearchCV")

                    # Log best parameters if available
                    if hasattr(evaluator, 'best_params_'):
                        for param_name, param_value in evaluator.best_params_.items():
                            mlflow.log_param(f"best_params.{param_name}", param_value)

        # Extract results for compatibility
        mean_rmse = cv_results['rmse_mean']
        std_rmse = cv_results['rmse_std']
        mean_r2 = cv_results['r2_mean']
        std_r2 = cv_results['r2_std']

        # Add MLflow model and results logging (Workshop 4)
        if mlflow.active_run():
            # Log cross-validation results
            mlflow.log_metric("cv.rmse_mean", mean_rmse)
            mlflow.log_metric("cv.rmse_std", std_rmse)
            mlflow.log_metric("cv.r2_mean", mean_r2)
            mlflow.log_metric("cv.r2_std", std_r2)
            
            # Log the trained model (for Model column)
                # Step 1: Prepare clean data for MLflow (avoid warnings)
                # Remove rows with missing values and convert to float64
            X_clean = X.dropna()
            y_clean = y[X_clean.index]

                # Step 2: Create MLflow model signature using clean data
                # The signature describes input/output format for the model
            from mlflow.models.signature import infer_signature
            signature = infer_signature(X_clean, y_clean)

                # Step 3: Create descriptive model name (appears in Model column)
            input_example = X_clean.iloc[:5]

                # Step 4: Prepare input example for MLflow documentation
                # This shows users what kind of data the model expects
            model_name_mlflow = f"{args.model}_cv_model"

                # Log model to MLflow
            mlflow.sklearn.log_model(
                model,
                model_name_mlflow,
                signature=signature,
                input_example=input_example,
                registered_model_name=f"air-quality-{args.model}"
            )

                # Register model in MLflow Model Registry
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_name_mlflow}"
            mlflow.register_model(model_uri, f"air-quality-{args.model}")

        # Step 5: Save Model (Workshop 4)
            # Retrain final model on all training data (best practice)
            with logger.timer("Retraining on all data"):
                # Prepare all training data
                X_full = train_features[selected_features].dropna()
                y_full = train_features[TARGET_COL][X_full.index]

                # Retrain model
                final_model = trainer.create_model(args.model)
                final_model.fit(X_full, y_full)

            # Log the final model to MLflow
                # Prepare clean data for MLflow model signature
            signature = infer_signature(X_full, y_full)
            input_example = X_full.iloc[:5]

            # Create model name for MLflow
                # Log final model to MLflow
            mlflow.sklearn.log_model(
                final_model,
                "final_model",
                signature=signature,
                input_example=input_example,
                registered_model_name=f"air-quality-{args.model}-production"
            )

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

        # Add MLflow final results logging (Workshop 4)
        if mlflow.active_run():
            # Log final summary metrics
            mlflow.log_metric("final.rmse", mean_rmse)
            mlflow.log_metric("final.r2_score", mean_r2)
            mlflow.log_param("execution.time_seconds", int(end_time - start_time))

            # Log results summary as artifact
            mlflow.log_dict(summary, "results_summary.json")

        logger.pipeline_complete(end_time - start_time)

        return {
            'results': results,
            'summary': summary,
            'selected_features': selected_features,
            'execution_time': execution_time
        }

    finally:
        # End MLflow run (Workshop 4)
        if mlflow.active_run():
            mlflow.end_run()

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

    # Add MLflow tracking argument --mlflow (Workshop 4)
    parser.add_argument(
        '--mlflow', action='store_true',
        help='Use mlflow dashboard for history'
    )
    
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
