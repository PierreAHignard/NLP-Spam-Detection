import subprocess
import os
import sys

# Define the relative path to your script from the parent directory
PIPELINE_SCRIPT = "scripts/run_pipeline.py"

# Define your scenarios
experiments = [
    {
        "name": "Baseline_Logistic_Email",
        "args": {
            "--train_datasets": ["EMAIL"],
            "--test_datasets": ["EMAIL"],
            "--model": "logistic",
        }
    },
    {
        "name": "Full_SMS_Cleaning",
        "args": {
            "--train_datasets": ["SMS"],
            "--test_datasets": ["SMS"],
            "--model": "logistic",  # Change to other models defined in your config
            "--lowercase": True,
            "--remove_punctuation": True,
            "--number_placeholder": True
        }
    },
    {
        "name": "Comparison_Run",
        "args": {
            "--compare": True,
            "--train_datasets": ["SMS", "EMAIL"],
            "--test_datasets": ["SMS", "EMAIL"],
            "--log-level": "verbose"
        }
    }
]


def run_experiments():
    # 1. Verify we are in the correct directory
    if not os.path.exists(PIPELINE_SCRIPT):
        print(f"❌ Error: Could not find '{PIPELINE_SCRIPT}'.")
        print("Please run this script from the parent directory (project root).")
        print(f"Current working directory: {os.getcwd()}")
        sys.exit(1)

    print(f"🚀 Starting automation sequence on {len(experiments)} experiments...\n")

    for i, exp in enumerate(experiments, 1):
        print(f"[{i}/{len(experiments)}] Running: {exp['name']}...")

        # Build command: uv run scripts/run_pipeline.py ...
        cmd = ["uv", "run", PIPELINE_SCRIPT]

        # Add run_name
        cmd.extend(["--run_name", exp["name"]])

        # Parse arguments dictionary into command list
        for key, value in exp["args"].items():
            if isinstance(value, bool):
                if value: cmd.append(key)  # Add flag only if True
            elif isinstance(value, list):
                cmd.append(key)
                cmd.extend(value)  # Add multiple values for nargs='+'
            else:
                cmd.append(key)
                cmd.append(str(value))

        try:
            # Run the command and wait for it to finish
            subprocess.run(cmd, check=True)
            print(f"✅ Finished: {exp['name']}\n")
        except subprocess.CalledProcessError:
            print(f"🚨 Failed: {exp['name']}. Continuing to next...\n")


if __name__ == "__main__":
    run_experiments()