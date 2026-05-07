import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Define the minimalist, modern color palette
COLOR_PRIMARY = "#2B2D42"   # Deep Indigo (for primary data/lines)
COLOR_SECONDARY = "#00A896" # Bright Cyan (for accents/gradients)
COLOR_ACCENT = "#EF233C"    # Crimson Red (for warnings/critical errors)
COLOR_TEXT = "#333333"      # Soft black for readability
COLOR_GRID = "#E0E0E0"      # Very subtle gray for grid lines

# Create a continuous custom colormap for gradients
gradient_cmap = LinearSegmentedColormap.from_list(
    "IndigoCyan", [COLOR_PRIMARY, COLOR_SECONDARY]
)

# Categorical palette for bar charts and distinct categories
categorical_palette = [COLOR_PRIMARY, COLOR_SECONDARY, "#4A4E69", "#9A8C98", "#C9ADA7"]

def apply_minimalist_theme():
    """Applies a minimalist, 'Tufte-inspired' modern academic theme."""
    # Reset to default to clear any existing configurations
    plt.style.use('default')
    
    # Typography
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Inter', 'Roboto', 'Helvetica Neue', 'Arial', 'sans-serif']
    plt.rcParams['text.color'] = COLOR_TEXT
    plt.rcParams['axes.labelcolor'] = COLOR_TEXT
    plt.rcParams['xtick.color'] = COLOR_TEXT
    plt.rcParams['ytick.color'] = COLOR_TEXT
    
    # Sizing for readability
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlepad'] = 20
    
    # Clean Backgrounds
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    
    # Spine (Border) Minimalism
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.left'] = True
    plt.rcParams['axes.spines.bottom'] = True
    plt.rcParams['axes.edgecolor'] = COLOR_GRID
    plt.rcParams['axes.linewidth'] = 1.2
    
    # Grid Styling (Soft horizontal lines only)
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.4
    plt.rcParams['grid.color'] = COLOR_GRID
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['axes.grid.axis'] = 'y' # Only horizontal lines by default
    
    # Output Quality
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    
    # Apply to seaborn
    sns.set_palette(categorical_palette)

