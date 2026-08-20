import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

# Set up the figure
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111)
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
fig.suptitle('Quality-Aware Explainable AI-Driven Multi-Gas Fusion with Uncertainty-Guided Transfer Learning\nFor Ketosis Detection via Breath Analysis',
             fontsize=16, fontweight='bold', y=0.95)

# Define box drawing function
def draw_box(ax, xy, width, height, text, fontsize=9, boxstyle="round,pad=0.3",
             fc="lightblue", ec="navy", lw=1.5):
    box = FancyBboxPatch(xy, width, height, boxstyle=boxstyle,
                         fc=fc, ec=ec, lw=lw)
    ax.add_patch(box)
    ax.text(xy[0] + width/2, xy[1] + height/2, text,
            ha='center', va='center', fontsize=fontsize, wrap=True, weight='bold')

# Define arrow function
def draw_arrow(ax, start, end, color='black', lw=1.5, style='->', connectionstyle="arc3,rad=0.0"):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, lw=lw, color=color,
                                connectionstyle=connectionstyle))

# ========== TOP LAYER: DATA SOURCES ==========
# Synthetic Data
draw_box(ax, (0.5, 8), 2.5, 1.2, 'Synthetic Multi-Gas Data\n• 5 gases: H₂, CH₄, H₂S, NH₃, Acetone\n• 5 quality vars: CO₂, Flow, Humidity, Temp, Stability\n• Simulated distortions: noise, drift, cross-sensitivity, poor capture',
         fontsize=8, fc="lightgreen", ec="darkgreen")

# Real Data (PeerJ)
draw_box(ax, (11, 8), 2.5, 1.2, 'Real PeerJ Breath-Blood Data\n• 1,214 samples\n• 19 subjects\n• Breath acetone (ppm)\n• Blood β-HB (mM)\n• No multi-gas or quality vars',
         fontsize=8, fc="lightcoral", ec="darkred")

# ========== MIDDLE LAYER: PREPROCESSING ==========
# Data Processing
draw_box(ax, (5.5, 8), 2.5, 1.2, 'Data Processing & Feature Engineering\n• Raw signals → 4 features:\n  - Raw breath PPM\n  - Rolling mean (quality)\n  - Rolling std (stability)\n  - 1st difference (dynamics)\n• Standardization per training fold',
         fontsize=8, fc="lightyellow", ec="gold")

# ========== PROCESSING LAYER ==========
# Quality Assessment
draw_box(ax, (2, 5.5), 3, 1.2, 'Quality Assessment\n• Breath Quality Score (BQS)\n• BQS = 0.35·QCO₂ + 0.20·QF + 0.15·QH + 0.10·QT + 0.20·QS\n• Quality indicators: CO₂, Flow, Humidity, Temp, Stability',
         fontsize=8, fc="lightblue", ec="navy")

# Adaptive Fusion (HBKI)
draw_box(ax, (6.5, 5.5), 3, 1.2, 'Adaptive Bayesian Ridge (HBKI)\n• Learns sample-specific feature weights\n• Uncertainty-aware attention\n• Outputs: Prediction (μ) + Uncertainty (σ)\n• Features: [H₂, CH₄, H₂S, NH₃, Acetone, CO₂, Flow, Humidity, Temp, Stability, BQS]',
         fontsize=8, fc="lightblue", ec="navy")

# ========== TRANSFER LAYER ==========
# Subject Loop Indicator
draw_box(ax, (0.5, 3), 2.5, 1.2, 'Leave-One-Subject-Out CV\nFor each test subject t:\n• Train on all other subjects\n• Predict for subject t\n• Aggregate metrics',
         fontsize=8, fc="lightpurple", ec="purple")

# Uncertainty-Guided Transfer Learning (NOVELTY)
draw_box(ax, (5.5, 3), 5.5, 1.8, 'UNCERTAINTY-GUIDED TRANSFER LEARNING (NOVELTY)\n\nFOR EACH TEST SAMPLE FROM SUBJECT t:\n1. Compute feature vector zₜ\n2. For each source subject s:\n   • Similarity: sₛ = exp(-||zₜ - μₛ||² / (2bw²))\n   • Reliability: rₛ = 1 / (σₛ + ε)  ← KEY NOVELTY\n   • Weight: wₛ = sₛ × rₛ\n3. Prediction: ŷₜ = Σ(wₛ · ŷₛ) / Σwₛ\n\nWHY IT WORKS:\n• Low-uncertainty subjects = reliable representatives → HIGHER weight\n• High-uncertainty subjects = outliers/noisy → LOWER weight\n• Reduces inter-subject variability by adapting to subject-specific reliability',
         fontsize=8, fc="lightpink", ec="maroon", lw=2)

# ========== OUTPUT LAYER ==========
# Final Prediction
draw_box(ax, (10.5, 3), 2.5, 1.2, 'Final Output\n• β-HB concentration estimate\n• Prediction uncertainty (σ)\n• Confidence intervals\n• Calibrated uncertainty estimates',
         fontsize=8, fc="lightgray", ec="black")

# ========== EVALUATION LAYER ==========
# Metrics Calculation
draw_box(ax, (5.5, 0.5), 5.5, 1.8, 'Evaluation Metrics\n• R²: Coefficient of determination (accuracy)\n• ECE: Expected Calibration Error (uncertainty quality)\n• 90% Coverage: Prediction interval reliability\n• Ablation Studies:\n  - Standard LOTO baseline\n  - Standard transfer learning\n  - UGTL similarity-only\n  - UGTL feature-gating-only\n  - Full UGTL (similarity + gating)\n• SHAP Explainability: Feature contribution analysis',
         fontsize=8, fc="lightyellow", ec="gold", lw=1.5)

# ========== ARROWS SHOWING FLOW ==========
# Data to Processing
draw_arrow(ax, (1.75, 8.6), (4.25, 8.6))  # Synthetic to Processing
draw_arrow(ax, (12.25, 8.6), (6.75, 8.6)) # Real to Processing

# Processing to Quality Assessment & HBKI
draw_arrow(ax, (6.75, 8.0), (3.5, 6.1))   # Processing to Quality Assessment
draw_arrow(ax, (6.75, 8.0), (8.0, 6.1))   # Processing to HBKI

# Subject loop connections
draw_arrow(ax, (1.75, 5.5), (1.75, 4.2))  # Quality Assessment to Subject Loop
draw_arrow(ax, (8.0, 5.5), (8.25, 4.2))   # HBKI to Subject Loop

# Subject Loop to UGTL
draw_arrow(ax, (1.75, 3.0), (3.0, 3.9))   # Subject Loop to UGTL

# UGTL to Final Output
draw_arrow(ax, (8.25, 3.9), (11.75, 3.6)) # UGTL to Final Output

# Final Output to Evaluation
draw_arrow(ax, (11.75, 3.0), (8.25, 1.7)) # Final Output to Evaluation

# Evaluation metrics notes
ax.text(0.5, 0.2, 'Key Innovation: Uncertainty estimates guide transfer learning weighting → reduces inter-subject variability',
        fontsize=10, style='italic', weight='bold', color='darkred')

# Add legend-like explanations
ax.text(0.5, 9.2, 'Data Sources', fontsize=11, weight='bold')
ax.text(0.5, 6.7, 'Core Processing', fontsize=11, weight='bold')
ax.text(0.5, 4.2, 'Cross-Subject Adaptation', fontsize=11, weight='bold')
ax.text(0.5, 1.2, 'Output & Validation', fontsize=11, weight='bold')

plt.tight_layout()
plt.savefig('C:\\Users\\priti\\OneDrive\\Desktop\\capstone\\research_paper\\results\\figures\\methodology_overview.png',
            dpi=300, bbox_inches='tight')
plt.close()

print("Methodology overview figure saved to: results/figures/methodology_overview.png")