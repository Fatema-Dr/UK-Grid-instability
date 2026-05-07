from PIL import Image, ImageDraw
import os

def sanitize_figure(path, top_pixels):
    print(f"Sanitizing {path}...")
    try:
        img = Image.open(path)
        width, height = img.size
        draw = ImageDraw.Draw(img)
        # Draw a white rectangle over the top area where the title is
        draw.rectangle([0, 0, width, top_pixels], fill="white")
        img.save(path)
        print(f"  Removed top {top_pixels} pixels.")
    except Exception as e:
        print(f"  Error processing {path}: {e}")

fig_dir = "/home/yogipatel/Documents/UK-Grid-instability/report/figures/"

# Specific pixel counts based on visual inspection
tasks = [
    ("figure_3_1_system_architecture.png", 220),
    ("figure_4_4_residual_analysis.png", 110),
    ("figure_4_2_feature_importance.png", 100),
    ("figure_5_4_calibration_reliability.png", 100),
    ("figure_5_2_shap_waterfall.png", 100),
    ("figure_5_3_intervention_simulation.png", 200),
    ("figure_5_5_shap_summary_beeswarm.png", 100),
    ("impressive_uncertainty_ribbon.png", 120)
]

for filename, pixels in tasks:
    path = os.path.join(fig_dir, filename)
    if os.path.exists(path):
        sanitize_figure(path, pixels)
    else:
        print(f"  File not found: {path}")
