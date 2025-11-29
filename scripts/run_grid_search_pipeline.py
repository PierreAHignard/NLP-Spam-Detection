import subprocess
import itertools
import os
import sys

PIPELINE_SCRIPT = "scripts/run_pipeline.py"

# Define parameters to sweep
param_grid = {
    "--model": ["Logistic_Regression", "Multinomial_NB", "Linear_SVC", "SGD_Classifier", "Random_Forest"],
    "--train_datasets": [["EMAIL"], ["SMS"], ["EMAIL", "SMS"]],
    "--test_datasets": [["EMAIL"], ["SMS"], ["EMAIL", "SMS"]],
    # Booleans to toggle
    "--optimize": [True, False],
    "--lowercase": [True],
    "--remove_punctuation": [True],
    "--stop_words": ["nltk", "None"],
    "--number_placeholder": [True],
    "--vectorizer_type": ['count', 'tfidf']
}


def run_grid_search():
    if not os.path.exists(PIPELINE_SCRIPT):
        print(f"❌ Error: Could not find '{PIPELINE_SCRIPT}'. Run from project root.")
        sys.exit(1)

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    print(f"🔍 Found {len(combinations)} combinations to test.\n")

    for i, combo in enumerate(combinations, 1):
        cmd = ["uv", "run", PIPELINE_SCRIPT]

        # Add fixed/static arguments here
        cmd.extend(["--log-level", "silent"])

        run_name_parts = ["Grid"]

        # Build dynamic arguments
        for key, value in zip(keys, combo):
            if isinstance(value, bool):
                if value:
                    cmd.append(key)
                    run_name_parts.append(key.strip('--'))
            elif isinstance(value, list):
                cmd.append(key)
                cmd.extend(value)
                run_name_parts.append(f"Train{''.join(value)}")
            else:
                cmd.append(key)
                cmd.append(str(value))
                run_name_parts.append(str(value))

        # Construct a readable run name
        run_name = "_".join(run_name_parts)
        cmd.extend(["--run_name", run_name])

        print(f"[{i}/{len(combinations)}] Executing: {run_name}")
        subprocess.run(cmd)


if __name__ == "__main__":
    run_grid_search()