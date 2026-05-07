import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Configuration for Dark Corporate Style
BG_COLOR = "#1e1e1e"  # Deep charcoal
BOX_COLOR = "#2d2d2d" # Slightly lighter
TEXT_COLOR = "#e0e0e0" # Soft white
ACCENT_BLUE = "#00bcd4" # Cyan accent
ACCENT_ORANGE = "#ff9800" # Orange accent
ACCENT_GREEN = "#4caf50" # Green accent
BORDER_COLOR = "#444444"

plt.rcParams.update({
    "text.color": TEXT_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "font.family": "sans-serif",
})

def create_system_architecture(output_path):
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Layers
    layers = [
        {"name": "DATA INGESTION", "y": 85, "color": ACCENT_BLUE, "items": ["NESO API (1s Freq)", "Open-Meteo (Weather)"]},
        {"name": "PHYSICS ENGINE", "y": 60, "color": ACCENT_ORANGE, "items": ["Polars Temporal Join", "RoCoF & OpSDA", "Inertia Proxies"]},
        {"name": "PREDICTIVE CORE", "y": 35, "color": ACCENT_GREEN, "items": ["LightGBM Quantiles", "LSTM Anomaly Monitor", "SHAP Explanations"]},
        {"name": "OPERATOR DASHBOARD", "y": 10, "color": "#f44336", "items": ["Real-time Alerting", "Intervention Sim"]},
    ]

    for layer in layers:
        # Layer Box
        rect = patches.FancyBboxPatch((5, layer['y'] - 5), 90, 15, boxstyle="round,pad=2", 
                                     linewidth=1, edgecolor=layer['color'], facecolor=BOX_COLOR, alpha=0.8)
        ax.add_patch(rect)
        
        # Layer Title
        ax.text(50, layer['y'] + 8, layer['name'], ha='center', va='center', fontweight='bold', fontsize=14, color=layer['color'])
        
        # Items
        item_str = " | ".join(layer['items'])
        ax.text(50, layer['y'] + 2, item_str, ha='center', va='center', fontsize=10)

        # Arrows (connecting to the layer below)
        if layer['name'] != "OPERATOR DASHBOARD":
            ax.annotate("", xy=(50, layer['y'] - 8), xytext=(50, layer['y'] - 1),
                        arrowprops=dict(arrowstyle="->", color=BORDER_COLOR, lw=2))

    plt.title("GridGuardian: Integrated System Architecture", color=TEXT_COLOR, fontsize=18, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=BG_COLOR)
    plt.close()

def create_alert_fusion_diagram(output_path):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Signal Groups
    ax.text(20, 85, "PHYSICAL SIGNALS", ha='center', fontweight='bold', color=ACCENT_ORANGE)
    phys_signals = ["RoCoF", "RoCoF Accel", "Volatility", "Renewable Stress", "Boundary Breach"]
    for i, s in enumerate(phys_signals):
        ax.text(20, 75 - i*8, f"● {s}", ha='left', fontsize=10)

    ax.text(50, 85, "ML SIGNALS", ha='center', fontweight='bold', color=ACCENT_BLUE)
    ml_signals = ["LGBM Prob > 0.3", "LSTM Prob > 0.3", "Quantile Breach"]
    for i, s in enumerate(ml_signals):
        ax.text(50, 75 - i*8, f"● {s}", ha='left', fontsize=10)

    # Fusion Box
    rect = patches.FancyBboxPatch((20, 20), 60, 15, boxstyle="round,pad=2", 
                                 linewidth=2, edgecolor=TEXT_COLOR, facecolor=BOX_COLOR)
    ax.add_patch(rect)
    ax.text(50, 27, "MULTI-PHYSICS FUSION LOGIC", ha='center', fontweight='bold', fontsize=12)
    ax.text(50, 23, "Weighted Signal Count (Σ)", ha='center', fontsize=10)

    # Output States
    ax.text(15, 5, "NORMAL", ha='center', fontweight='bold', color=ACCENT_GREEN)
    ax.text(50, 5, "CAUTION (Σ ≥ 2)", ha='center', fontweight='bold', color=ACCENT_ORANGE)
    ax.text(85, 5, "CRITICAL (Σ ≥ 3)", ha='center', fontweight='bold', color="#f44336")

    # Arrows - refined
    ax.annotate("", xy=(50, 36), xytext=(35, 55), arrowprops=dict(arrowstyle="->", color=BORDER_COLOR, alpha=0.6))
    ax.annotate("", xy=(50, 36), xytext=(65, 55), arrowprops=dict(arrowstyle="->", color=BORDER_COLOR, alpha=0.6))
    ax.annotate("", xy=(50, 12), xytext=(50, 19), arrowprops=dict(arrowstyle="->", color=BORDER_COLOR, alpha=0.8))
    
    plt.title("Multi-Physics Alert Fusion Strategy", color=TEXT_COLOR, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=BG_COLOR)
    plt.close()

def create_data_journey(output_path):
    fig, ax = plt.subplots(figsize=(12, 4), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    steps = [
        {"name": "INGEST", "desc": "NESO & Weather APIs", "color": ACCENT_BLUE},
        {"name": "REFINE", "desc": "Polars Join & Physics", "color": ACCENT_ORANGE},
        {"name": "PREDICT", "desc": "Quantile Ensemble", "color": ACCENT_GREEN},
        {"name": "EXPLAIN", "desc": "SHAP Risk Drivers", "color": ACCENT_BLUE},
        {"name": "ACTION", "desc": "Control Room Response", "color": "#f44336"}
    ]

    for i, step in enumerate(steps):
        x = 10 + i * 20
        # Circle
        circle = patches.Circle((x, 60), 8, color=step['color'], alpha=0.8)
        ax.add_patch(circle)
        ax.text(x, 60, str(i+1), ha='center', va='center', fontweight='bold', color='white', fontsize=14)
        
        # Labels
        ax.text(x, 40, step['name'], ha='center', fontweight='bold', color=step['color'], fontsize=12)
        ax.text(x, 30, step['desc'], ha='center', fontsize=9, color=TEXT_COLOR)

        # Connection lines
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 12, 60), xytext=(x + 8, 60), 
                        arrowprops=dict(arrowstyle="->", color=BORDER_COLOR, lw=2))

    plt.title("The Data Journey: From Raw Streams to Grid Safety", color=TEXT_COLOR, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=BG_COLOR)
    plt.close()

if __name__ == "__main__":
    output_dir = "/home/fatema/University/Dissertation/Final Report/figures"
    os.makedirs(output_dir, exist_ok=True)
    
    create_system_architecture(os.path.join(output_dir, "premium_architecture_dark.png"))
    create_alert_fusion_diagram(os.path.join(output_dir, "premium_alert_fusion.png"))
    create_data_journey(os.path.join(output_dir, "premium_data_journey.png"))
    print("✅ All premium diagrams generated in dark mode corporate style.")
