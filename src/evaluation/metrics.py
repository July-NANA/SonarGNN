import os
import pickle
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.utils.plot import plot_lr_comparison


def load_results(lr):
    """
    Load experimental results for a given learning rate.

    Args:
        lr (float): Learning rate value used in the experiment.

    Returns:
        dict or None: The loaded metrics dictionary if the pickle file exists,
                      otherwise None.
    """
    result_dir = f"results/lr_experiments/lr_{lr}"
    metrics_path = os.path.join(result_dir, "metrics.pkl")

    if os.path.exists(metrics_path):
        with open(metrics_path, "rb") as f:
            return pickle.load(f)
    return None


def main():
    """
    Main function: loads results for multiple learning rates and generates a
    comparison plot.
    """
    learning_rates = [0.1, 0.01, 0.001]
    results = {}

    # Iterate over each learning rate and attempt to load its metrics
    for lr in learning_rates:
        metrics = load_results(lr)
        if metrics:
            results[lr] = metrics
        else:
            # Print a warning if no results were found for this learning rate
            print(f"Warning: No results found for LR={lr}")

    # If at least one set of results was loaded successfully, generate the comparison plot
    if results:
        plot_lr_comparison(results, "results/lr_comparison.png")
    else:
        print("No valid results to plot")


if __name__ == "__main__":
    main()
