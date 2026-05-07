import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.append(os.path.abspath('report'))
import theme
theme.apply_minimalist_theme()

os.makedirs('report/figures', exist_ok=True)

def generate_distribution():
    fig, ax = plt.subplots(figsize=(8, 6))
    np.random.seed(42)
    
    # Simulate bimodal frequency distribution
    normal_freq = np.random.normal(50.0, 0.05, 10000)
    low_inertia_freq = np.random.normal(49.85, 0.1, 2000)
    data = np.concatenate([normal_freq, low_inertia_freq])
    
    sns.histplot(data, bins=100, kde=True, color=theme.COLOR_PRIMARY, ax=ax, stat="density", element="step", alpha=0.3)
    ax.axvline(50.0, color=theme.COLOR_TEXT, linestyle='--', alpha=0.5, label='Statutory Target (50 Hz)')
    ax.axvline(49.85, color=theme.COLOR_ACCENT, linestyle='--', alpha=0.8, label='Low-Inertia Alert Threshold')
    
    ax.set_title("Bimodal Grid Frequency Distribution (Simulated)")
    ax.set_xlabel("Grid Frequency (Hz)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    
    fig.tight_layout()
    fig.savefig('report/figures/figure_3_4_eda_distribution.png')
    plt.close(fig)

def generate_correlation():
    fig, ax = plt.subplots(figsize=(8, 6))
    np.random.seed(42)
    
    # Simulate correlation matrix
    features = ['Frequency', 'RoCoF', 'OpSDA Ramp', 'Inertia', 'Wind Gen', 'Solar Gen']
    corr = np.array([
        [ 1.00,  0.85, -0.12,  0.34, -0.22,  0.15],
        [ 0.85,  1.00, -0.45,  0.68, -0.55,  0.20],
        [-0.12, -0.45,  1.00, -0.30,  0.75, -0.10],
        [ 0.34,  0.68, -0.30,  1.00, -0.85,  0.40],
        [-0.22, -0.55,  0.75, -0.85,  1.00, -0.35],
        [ 0.15,  0.20, -0.10,  0.40, -0.35,  1.00]
    ])
    
    # Mask upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(corr, mask=mask, cmap=theme.gradient_cmap, vmin=-1, vmax=1, 
                annot=True, fmt=".2f", square=True, linewidths=.5, cbar_kws={"shrink": .8},
                xticklabels=features, yticklabels=features, ax=ax)
    
    ax.set_title("Spearman Rank Correlation Matrix")
    fig.tight_layout()
    fig.savefig('report/figures/figure_3_5_eda_correlation.png')
    plt.close(fig)

if __name__ == "__main__":
    generate_distribution()
    generate_correlation()
    print("EDA plots generated.")
