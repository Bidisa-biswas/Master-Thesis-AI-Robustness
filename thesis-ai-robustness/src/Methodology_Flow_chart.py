import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(14, 22))
ax.set_xlim(0, 12)
ax.set_ylim(0, 45)
ax.axis('off')

# Helper function to draw boxes
def draw_box(x, y, width, height, text, color='lightblue', fontsize=8):
    rect = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.05",
                                   facecolor=color, edgecolor='black', linewidth=1)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=fontsize, wrap=True)

# Helper function to draw arrows
def draw_arrow(x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='black', lw=1))

# ============================================================
# TITLE
# ============================================================
draw_box(2, 42, 8, 1, "LSTM VOLATILITY PREDICTION & STRESS TESTING\nMETHODOLOGY FLOWCHART", color='lightgray', fontsize=10)

# ============================================================
# STEP 1: DATA COLLECTION
# ============================================================
draw_box(2, 38, 8, 1.5, "DATA COLLECTION & PREPARATION\nS&P 500 (1997-2024) Daily closing prices", color='lightblue', fontsize=8)
draw_arrow(6, 38, 6, 36.5)

draw_box(2, 34, 8, 1.2, "LOG RETURNS CALCULATION\nr = ln(P_t/P_{t-1}) x 100", color='lightblue', fontsize=8)
draw_arrow(6, 34, 6, 32.5)

draw_box(2, 30, 8, 1.2, "TRAIN/VALIDATION SPLIT\nTrain: 2010-2019 (2,516 days) | Validation: 20%", color='lightblue', fontsize=8)
draw_arrow(6, 30, 6, 28.5)

# ============================================================
# STEP 2: VOLATILITY LABELS
# ============================================================
draw_box(2, 25, 8, 1.5, "CREATE VOLATILITY LABELS\ny = 1 if |r_{t+1}| > (1/20)S|r_{t-i}|\n1=HIGH (40.4%), 0=LOW (59.6%)", color='lightgreen', fontsize=7)
draw_arrow(6, 25, 6, 23.5)

draw_box(2, 21.5, 8, 1.2, "CREATE SEQUENCES (60 DAYS)\nX = [r_{t-59},...,r_t] -> y = label_{t+1}\nX shape: (2,455, 60)", color='lightgreen', fontsize=7)
draw_arrow(6, 21.5, 6, 20)

# ============================================================
# STEP 3: LSTM ARCHITECTURE
# ============================================================
draw_box(2, 16, 8, 2.5, "LSTM MODEL ARCHITECTURE\nLayer 1: LSTM(32, return_sequences=False) -> 4,352 params\nLayer 2: Dropout(0.2)\nLayer 3: Dense(16, ReLU) -> 528 params\nLayer 4: Dense(1, Sigmoid) -> 17 params\nOptimizer: Adam (lr=0.001), Loss: Binary cross-entropy", color='lightyellow', fontsize=7)
draw_arrow(6, 16, 6, 14.5)

# ============================================================
# STEP 4: TRAINING
# ============================================================
draw_box(2, 11.5, 8, 1.5, "MODEL TRAINING\nEpochs = 100, Batch size = 32\nClass weights: balanced (0:0.84, 1:1.48)\nCallbacks: EarlyStopping, ReduceLROnPlateau", color='lightyellow', fontsize=7)
draw_arrow(6, 11.5, 6, 10.5)

# ============================================================
# STEP 5: VALIDATION
# ============================================================
draw_box(2, 8.5, 8, 1.2, "VALIDATION PERFORMANCE\nValidation accuracy: 66.80%\nBalanced accuracy: 64.76%\n[OK] Outperforms naive baseline (59.6%)", color='lightcoral', fontsize=7)
draw_arrow(6, 8.5, 6, 7.5)

# ============================================================
# STEP 6: GARCH & MONTE CARLO
# ============================================================
draw_box(2, 5, 8, 1.5, "GARCH-MONTE CARLO STRESS SCENARIOS\n6 scenarios x 10,000 paths x 252 days\nNormal, 2008, COVID, Synthetic, Squeeze, Structural Break", color='lightblue', fontsize=7)
draw_arrow(6, 5, 6, 4)

# ============================================================
# STEP 7: RESULTS
# ============================================================
draw_box(2, 1.5, 8, 1.2, "RESULTS & REGULATORY MAPPING\nNormal: 50.73%, Non-linear Squeeze: 51.81% (p<0.001)\nEU AI Act thresholds: <10% Compliant, >30% Non-compliant", color='lightgreen', fontsize=7)

# ============================================================
# SIDE BOX: HYPERPARAMETER SENSITIVITY
# ============================================================
draw_box(11, 30, 3, 3, "HYPERPARAMETER SENSITIVITY\nLookback: 30/60/90 days\nLearning rate: 0.001/0.0005/0.0001\nDropout: 0.2/0.3/0.4", color='lightgray', fontsize=6)

plt.tight_layout()
plt.savefig('thesis_flowchart.png', dpi=150, bbox_inches='tight')
plt.show()