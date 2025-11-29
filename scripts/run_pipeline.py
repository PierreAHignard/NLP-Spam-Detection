#!/usr/bin/env python3
"""
Simple Air Quality ML Pipeline with Inline MLflow Integration

This pipeline includes MLflow logging directly in the main workflow without
utility functions, making it easy for students to understand.
"""

import sys
import time
from pathlib import Path
from sklearn.exceptions import ConvergenceWarning
import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pipeline import DataProcessor, FeatureEngineer, ModelTrainer
from pipeline.evaluator import hyperparameter_optimization_cv, calculate_metrics
from utils.config import MODEL_TYPES, DEFAULT_PARAM_GRIDS, DATA_PATH, MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI

from utils.logger import get_logger, set_log_level, log_level_from_string, LogLevel
from utils.utils import format_time_elapsed
from utils.parse_args import parse_arguments

import mlflow
from mlflow.models.signature import infer_signature
import warnings

def run_pipeline(args):
    """
    Run the complete air quality prediction pipeline with inline MLflow integration.
    """
    start_time = time.time()
    logger = get_logger()

    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")

    #Reduce the amount of noise shown by mlflow
    warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")
    warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

    # Creating Eperiment name (easier to separate this way)
    experiment_name = "Spam Detection"
    for data in args.train_datasets:
        experiment_name += f"_{data}"

    experiment_name += "_to"

    for data in args.test_datasets:
        experiment_name += f"_{data}"

    # Configuration MLflow simple
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    # Create descriptive run name
    if args.run_name == '':
        run_name = f"spam_{args.model}"

        if args.optimize:
            run_name += "_opti"

    else:
       run_name = args.run_name

    timestamp = datetime.datetime.now().strftime("%Hh%M")

    mlflow.start_run(run_name=f"{run_name}_{timestamp}")

    # Set tags for Dataset and Model columns in MLflow UI
    # PARAMS
    mlflow.log_param("model.type", args.model)
    mlflow.log_param("model.optimize", args.optimize)

    # TAGS
    mlflow.set_tag("dataset.path", DATA_PATH)
    mlflow.set_tag("mlflow.note.content",
               f"Pipeline with {args.model} model.")

    run = mlflow.active_run()
    logger.info("Run started:", run is not None)

    try:
        # Pipeline header with configuration
        logger.header("SPAM DETECTION ML PIPELINE")

        with logger.indent():
            logger.info(f"Model: {args.model}")
            logger.info(f"Optimization: {'Enabled' if args.optimize else 'Disabled'}")

############################################### DATA PIPELINE ##########################################################

        # Create the dataset selection
        train_selection = list(args.train_datasets)
        test_selection = list(args.test_datasets)

        with logger.indent():
            logger.data_info(f"Train data : {train_selection}")
            logger.data_info(f"Test data : {test_selection}")

        if mlflow.active_run():
            mlflow.log_params({
                "train.sms": "SMS" in train_selection,
                "train.email": "EMAIL" in train_selection,
                "test.sms": "SMS" in test_selection,
                "test.email": "EMAIL" in test_selection,
            })

        # Initialize components
        processor = DataProcessor(train_selection, test_selection)
        engineer = FeatureEngineer(
            stop_words=args.stop_words,
            lowercase=args.lowercase,
            remove_punctuation=args.remove_punctuation,
            number_placeholder=args.number_placeholder,
            vectorizer_type=args.vectorizer_type,
            max_features=args.vocabulary_size
        )
        trainer = ModelTrainer()

        # Step 1: Data Loading and Preprocessing
        logger.step("Data Loading and Preprocessing", 1)
        with logger.timer("Data loading and preprocessing"):
            train_msg, train_lab, test_msg, test_lab = processor.load_and_preprocess(
                drop_duplicates=True,
                balance= not args.optimize # Shouldn't balance if there is cross-validation
            )

        # Step 2: Feature Engineering
        logger.step("Text Preprocessing and Tokenisation", 2)
        with logger.timer("Text Preprocessing and Tokenisation"):
            train_msg, test_msg = engineer.tokeniser(train_msg, test_msg)

############################################### MODEL PIPELINE #########################################################

        logger.step("Model Pipeline", 3)

        # Instance the variable -> will only be set if optimised
        best_params = None

        # Create model
        if args.model in MODEL_TYPES:
            model = trainer.create_model(args.model)
        else:
            raise Exception(f"The input model '{args.model}' is not recognised")

        if not args.optimize:
            # CASE 1 — no optimization
            with logger.timer("Model Training (no optimization)"):
                model = trainer.train_single_model(train_msg, train_lab, args.model)

        else:
            # CASE 2 — optimization requested
            # Check if the model has a grid
            param_grid = DEFAULT_PARAM_GRIDS.get(args.model, None)

            # If no grid, fallback to training on default values
            if param_grid is None:
                logger.warning(f"No parameter grid for '{args.model}'. Training will be performed on default values")

                with logger.timer("Model Training (no optimization)"):
                    model = trainer.train_single_model(train_msg, train_lab, args.model)

            else:
                # Grid exists -> perform optimization
                with logger.timer("Hyperparameter optimization"):
                    model, best_params, _ = hyperparameter_optimization_cv(model, param_grid, train_msg, train_lab)

################################################# SAVE MODEL ###########################################################

        # Step 6: Results Summary
        logger.step("Results Summary", 4)

        end_time = time.time()
        execution_time = format_time_elapsed(start_time, end_time)

        summary = {
            'Model': str(model),
            'Optimized': args.optimize,
            'Execution Time': execution_time,
            'training.metrics': calculate_metrics(model.predict(train_msg), train_lab),
            'testing.metrics': calculate_metrics(model.predict(test_msg), test_lab),
            'model.best_params': best_params or 'Not optimised'
        }

        logger.results_summary(summary)

        # Add MLflow final results logging
        if mlflow.active_run():
            # Log final summary metrics
            mlflow.log_metric("test.accuracy", summary["testing.metrics"]["accuracy"])
            mlflow.log_metric("test.precision", summary["testing.metrics"]["precision"])
            mlflow.log_metric("test.recall", summary["testing.metrics"]["recall"])
            mlflow.log_param("execution.time_seconds", int(end_time - start_time))

            # Log results summary as artifact
            mlflow.log_dict(summary, "results_summary.json")

            # Log Model
            mlflow.sklearn.log_model(
                model,
                signature=infer_signature(train_msg, train_lab),
                input_example=train_msg[:5],
                registered_model_name= experiment_name,
                tags={
                    "trained.sms": "SMS" in processor.train_selection,
                    "trained.email": "EMAIL" in processor.train_selection
                }
            )

        logger.pipeline_complete(end_time - start_time)

        return summary

    finally:
        # End MLflow run
        if mlflow.active_run():
            mlflow.end_run()


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
