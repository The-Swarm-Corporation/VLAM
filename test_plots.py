#!/usr/bin/env python3
"""
Simple test script to demonstrate the matplotlib plotting functionality
from the VLA benchmark suite.
"""

import matplotlib.pyplot as plt
import numpy as np
import os


def test_basic_plotting():
    """Test basic matplotlib functionality."""
    print("Testing basic matplotlib functionality...")

    # Create output directory
    os.makedirs("plots", exist_ok=True)

    # Sample data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)

    # Create a simple plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, y1, "b-", label="sin(x)", linewidth=2)
    plt.plot(x, y2, "r--", label="cos(x)", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Simple Trigonometric Functions")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save plot
    plot_path = "plots/test_basic_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Basic plot saved to: {plot_path}")

    plt.close()


def test_vla_style_plots():
    """Test VLA benchmark style plots."""
    print("Testing VLA benchmark style plots...")

    # Sample VLA benchmark data
    sequence_lengths = [10, 25, 50, 100, 200, 500]
    mamba_times = [2.1, 3.8, 6.2, 11.5, 22.1, 45.3]
    transformer_times = [2.5, 4.8, 9.2, 18.5, 37.1, 92.3]

    mamba_memory = [128, 145, 178, 234, 345, 567]
    transformer_memory = [135, 165, 220, 320, 480, 890]

    # Create performance comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("VLA Model Performance Demo", fontsize=16, fontweight="bold")

    colors = ["#1f77b4", "#ff7f0e"]
    markers = ["o", "s"]

    # Plot 1: Inference time scaling
    ax1.plot(
        sequence_lengths,
        mamba_times,
        marker=markers[0],
        label="Mamba VLA",
        color=colors[0],
        linewidth=2,
        markersize=8,
        alpha=0.8,
    )
    ax1.plot(
        sequence_lengths,
        transformer_times,
        marker=markers[1],
        label="Transformer VLA",
        color=colors[1],
        linewidth=2,
        markersize=8,
        alpha=0.8,
    )
    ax1.set_xlabel("Sequence Length", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Inference Time (ms)", fontsize=12, fontweight="bold")
    ax1.set_title("Inference Time Scaling", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor("#f8f9fa")

    # Plot 2: Memory usage scaling
    ax2.plot(
        sequence_lengths,
        mamba_memory,
        marker=markers[0],
        label="Mamba VLA",
        color=colors[0],
        linewidth=2,
        markersize=8,
        alpha=0.8,
    )
    ax2.plot(
        sequence_lengths,
        transformer_memory,
        marker=markers[1],
        label="Transformer VLA",
        color=colors[1],
        linewidth=2,
        markersize=8,
        alpha=0.8,
    )
    ax2.set_xlabel("Sequence Length", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Memory Usage (MB)", fontsize=12, fontweight="bold")
    ax2.set_title("Memory Usage Scaling", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor("#f8f9fa")

    # Plot 3: Performance comparison bar chart
    models = ["Mamba VLA", "Transformer VLA"]
    avg_times = [np.mean(mamba_times), np.mean(transformer_times)]
    avg_memory = [np.mean(mamba_memory), np.mean(transformer_memory)]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax3.bar(
        x - width / 2,
        avg_times,
        width,
        label="Inference Time (ms)",
        color=colors[0],
        alpha=0.8,
    )
    bars2 = ax3.bar(
        x + width / 2,
        avg_memory,
        width,
        label="Memory Usage (MB)",
        color=colors[1],
        alpha=0.8,
    )

    ax3.set_xlabel("Models", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Metrics", fontsize=12, fontweight="bold")
    ax3.set_title("Performance Overview", fontsize=14, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Speed vs memory trade-off
    ax4.scatter(
        avg_times,
        avg_memory,
        s=200,
        c=colors,
        alpha=0.7,
        edgecolors="black",
        linewidth=2,
    )

    for i, model in enumerate(models):
        ax4.annotate(
            model,
            (avg_times[i], avg_memory[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )

    ax4.set_xlabel("Inference Time (ms)", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Memory Usage (MB)", fontsize=12, fontweight="bold")
    ax4.set_title("Speed vs Memory Trade-off", fontsize=14, fontweight="bold")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    # Save plots
    plot_path = "plots/vla_test_plots.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"VLA style plots saved to: {plot_path}")

    # Also save as PDF
    pdf_path = "plots/vla_test_plots.pdf"
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    print(f"VLA style plots also saved as PDF: {pdf_path}")

    plt.close()


def test_heatmap():
    """Test heatmap creation."""
    print("Testing heatmap creation...")

    # Create sample data matrix
    models = ["Mamba VLA", "Transformer VLA"]
    batch_sizes = [1, 4, 8, 16]

    # Simulate memory usage matrix
    memory_matrix = np.array(
        [[128, 145, 178, 234], [135, 165, 220, 320]]  # Mamba VLA  # Transformer VLA
    )

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(memory_matrix, cmap="viridis", aspect="auto")

    # Set labels
    ax.set_xticks(range(len(batch_sizes)))
    ax.set_yticks(range(len(models)))
    ax.set_xticklabels(batch_sizes)
    ax.set_yticklabels(models)
    ax.set_xlabel("Batch Size", fontsize=12, fontweight="bold")
    ax.set_ylabel("Model", fontsize=12, fontweight="bold")
    ax.set_title("Memory Usage Heatmap (MB)", fontsize=14, fontweight="bold")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Memory Usage (MB)", fontsize=10)

    # Add text annotations
    for i in range(len(models)):
        for j in range(len(batch_sizes)):
            text = ax.text(
                j,
                i,
                f"{memory_matrix[i, j]:.0f}",
                ha="center",
                va="center",
                color="white",
                fontweight="bold",
            )

    plt.tight_layout()

    # Save heatmap
    plot_path = "plots/test_heatmap.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Heatmap saved to: {plot_path}")

    plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("MATPLOTLIB PLOTTING DEMONSTRATION")
    print("=" * 60)

    try:
        # Test basic plotting
        test_basic_plotting()

        # Test VLA style plots
        test_vla_style_plots()

        # Test heatmap
        test_heatmap()

        print("\n" + "=" * 60)
        print("ALL PLOTS CREATED SUCCESSFULLY!")
        print("Check the 'plots/' directory for generated files:")
        print("  - test_basic_plot.png")
        print("  - vla_test_plots.png")
        print("  - vla_test_plots.pdf")
        print("  - test_heatmap.png")
        print("=" * 60)

    except Exception as e:
        print(f"Error during plotting: {e}")
        raise
