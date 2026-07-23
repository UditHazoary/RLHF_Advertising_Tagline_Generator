import os

import matplotlib.pyplot as plt
import pandas as pd

# ==========================================================
# Paths
# ==========================================================

LOG_FILE = "logs/dpo_training_metrics.csv"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# Load Metrics
# ==========================================================

df = pd.read_csv(LOG_FILE)

# ==========================================================
# Helper Function
# ==========================================================

def save_plot(x, y, title, xlabel, ylabel, filename):
    plt.figure(figsize=(8,5))
    plt.plot(x, y, linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()


# ==========================================================
# Individual Plots
# ==========================================================

save_plot(
    df["step"],
    df["loss"],
    "Training Loss",
    "Training Step",
    "Loss",
    "loss_curve.png",
)

save_plot(
    df["step"],
    df["rewards/accuracies"],
    "Reward Accuracy",
    "Training Step",
    "Accuracy",
    "reward_accuracy.png",
)

save_plot(
    df["step"],
    df["rewards/margins"],
    "Reward Margin",
    "Training Step",
    "Margin",
    "reward_margin.png",
)

save_plot(
    df["step"],
    df["grad_norm"],
    "Gradient Norm",
    "Training Step",
    "Gradient Norm",
    "gradient_norm.png",
)

save_plot(
    df["step"],
    df["learning_rate"],
    "Learning Rate",
    "Training Step",
    "Learning Rate",
    "learning_rate.png",
)

# ==========================================================
# Chosen vs Rejected Rewards
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(
    df["step"],
    df["rewards/chosen"],
    label="Chosen Reward",
)

plt.plot(
    df["step"],
    df["rewards/rejected"],
    label="Rejected Reward",
)

plt.title("Chosen vs Rejected Rewards")
plt.xlabel("Training Step")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "chosen_vs_rejected_rewards.png"),
    dpi=300,
)

plt.close()

# ==========================================================
# Dashboard
# ==========================================================

fig, axes = plt.subplots(2, 3, figsize=(16, 9))

plots = [
    ("loss", "Loss"),
    ("rewards/accuracies", "Reward Accuracy"),
    ("rewards/margins", "Reward Margin"),
    ("grad_norm", "Gradient Norm"),
    ("learning_rate", "Learning Rate"),
]

positions = [
    (0,0),
    (0,1),
    (0,2),
    (1,0),
    (1,1),
]

for (column, title), (r,c) in zip(plots, positions):
    axes[r,c].plot(df["step"], df[column])
    axes[r,c].set_title(title)
    axes[r,c].grid(True)

axes[1,2].plot(df["step"], df["rewards/chosen"], label="Chosen")
axes[1,2].plot(df["step"], df["rewards/rejected"], label="Rejected")
axes[1,2].set_title("Reward Comparison")
axes[1,2].legend()
axes[1,2].grid(True)

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "training_dashboard.png"),
    dpi=300,
)

plt.close()

print("Training plots saved to outputs/")