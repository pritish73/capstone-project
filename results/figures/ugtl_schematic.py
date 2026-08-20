import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

# Set up the figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax1.axis('off')
ax2.axis('off')

# Panel A: Overall Framework
def draw_box(ax, xy, width, height, text, fontsize=10, boxstyle="round,pad=0.3"):
    box = FancyBboxPatch(xy, width, height, boxstyle=boxstyle,
                         fc="lightblue", ec="navy", lw=1.5)
    ax.add_patch(box)
    ax.text(xy[0] + width/2, xy[1] + height/2, text,
            ha='center', va='center', fontsize=fontsize, wrap=True)

# Boxes for overall framework
draw_box(ax1, (0.5, 7), 2, 1.5, 'Multi-Gas Sensors\n(5 gases + 5 quality vars)',
         fontsize=9)
draw_box(ax1, (3.5, 7), 2, 1.5, 'Quality Assessment\n(Breath Quality Score)',
         fontsize=9)
draw_box(ax1, (6.5, 7), 2, 1.5, 'Adaptive Fusion\n(HBKI)',
         fontsize=9)
draw_box(ax1, (8.5, 4), 1.5, 1.2, 'Prediction +\nUncertainty',
         fontsize=9)
draw_box(ax1, (6.5, 1), 2, 1.5, 'Uncertainty-Guided\nTransfer Learning',
         fontsize=9)
draw_box(ax1, (3.5, 1), 2, 1.5, 'Adapted Prediction\nfor Target Subject',
         fontsize=9)

# Arrows for overall framework
# Sensors -> Quality Assessment
ax1.annotate('', xy=(3.5, 7.75), xytext=(2.5, 7.75),
             arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
# Quality Assessment -> HBKI
ax1.annotate('', xy=(6.5, 7.75), xytext=(5.5, 7.75),
             arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))
# HBKI -> Prediction + Uncertainty
ax1.annotate('', xy=(9.25, 5.2), xytext=(7.5, 7.0),
             arrowprops=dict(arrowstyle='->', lw=1.5, color='black',
                             connectionstyle="arc3,rad=-0.3"))
# Prediction + Uncertainty -> UGTL
ax1.annotate('', xy=(7.5, 2.5), xytext=(8.5, 4.0),
             arrowprops=dict(arrowstyle='->', lw=1.5, color='black',
                             connectionstyle="arc3,rad=0.3"))
# UGTL -> Adapted Prediction
ax1.annotate('', xy=(4.5, 1.75), xytext=(7.5, 1.75),
             arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

# Panel B: UGTL Mechanism Details
# We'll show 3 source subjects contributing to 1 target
# Source subjects
source_y = [2, 5, 8]
for i, y in enumerate(source_y):
    draw_box(ax2, (1, y-0.4), 2, 0.8,
             f'Source Subject {i+1}\nFeature: z_{{i+1}}\nUncertainty: σ_{{i+1}}\nPrediction: ŷ_{{i+1}}',
             fontsize=8)

# Target subject
draw_box(ax2, (6, 4), 2, 0.8,
         'Target Subject\nFeature: z_t\nUncertainty: σ_t\nPrediction: ŷ_t',
         fontsize=8)

# Similarity and Reliability calculations
# Similarity: Gaussian kernel
ax2.text(3.5, 7, 'Similarity Calculation:', fontsize=9, weight='bold')
ax2.text(3.5, 6.5, 's_i = exp(-||z_t - z_i||² / (2σ²))', fontsize=8)
# Reliability: inverse uncertainty
ax2.text(3.5, 5.5, 'Reliability Calculation:', fontsize=9, weight='bold')
ax2.text(3.5, 5.0, 'r_i = 1 / (σ_i + ε)', fontsize=8)
# Weight combination
ax2.text(3.5, 4.0, 'Weight Calculation:', fontsize=9, weight='bold')
ax2.text(3.5, 3.5, 'w_i = s_i * r_i', fontsize=8)
ax2.text(3.5, 3.0, 'ŷ_t = Σ (w_i * ŷ_i) / Σ w_i', fontsize=8)

# Arrows from sources to calculations
for i, y in enumerate(source_y):
    # Similarity arrow
    ax2.annotate('', xy=(3.5, 6.8), xytext=(2, y),
                 arrowprops=dict(arrowstyle='->', lw=1, color='gray',
                                 connectionstyle="arc3,rad=0.2"))
    # Reliability arrow
    ax2.annotate('', xy=(3.5, 5.3), xytext=(2, y),
                 arrowprops=dict(arrowstyle='->', lw=1, color='gray',
                                 connectionstyle="arc3,rad=-0.2"))

# Arrows from calculations to target
# Similarity to weights
ax2.annotate('', xy=(4.5, 3.8), xytext=(3.5, 6.5),
             arrowprops=dict(arrowstyle='->', lw=1, color='black',
                             connectionstyle="arc3,rad=0.2"))
# Reliability to weights
ax2.annotate('', xy=(4.5, 3.8), xytext=(3.5, 5.0),
             arrowprops=dict(arrowstyle='->', lw=1, color='black',
                             connectionstyle="arc3,rad=-0.2"))
# Weights to target prediction
ax2.annotate('', xy=(7, 4.4), xytext=(5, 3.8),
             arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

# Title for Panel B
ax2.text(5, 9.5, 'Uncertainty-Guided Transfer Learning Mechanism',
         ha='center', fontsize=12, weight='bold')

plt.tight_layout()
plt.savefig('C:\\Users\\priti\\OneDrive\\Desktop\\capstone\\research_paper\\results\\figures\\ugtl_schematic.png',
            dpi=300, bbox_inches='tight')
plt.close()

print("Figure saved to: results/figures/ugtl_schematic.png")