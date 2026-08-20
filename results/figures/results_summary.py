import matplotlib.pyplot as plt
import numpy as np

# Create a bar chart comparing different approaches
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Synthetic Data Results
approaches_synthetic = ['Conventional\nGas Fusion', 'Quality-Aware\nFusion (HBKI)']
r2_synthetic = [0.892, 0.936]
ece_synthetic = [0.105, 0.082]
coverage_synthetic = [0.891, 0.948]

x = np.arange(len(approaches_synthetic))
width = 0.25

bars1 = ax1.bar(x - width, r2_synthetic, width, label='R²', color='skyblue', edgecolor='navy')
bars2 = ax1.bar(x, ece_synthetic, width, label='ECE (lower better)', color='lightcoral', edgecolor='darkred')
bars3 = ax1.bar(x + width, coverage_synthetic, width, label='90% Coverage', color='lightgreen', edgecolor='darkgreen')

ax1.set_xlabel('Approach', fontsize=12, fontweight='bold')
ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
ax1.set_title('Synthetic Data Performance\n(Proof of Concept)', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(approaches_synthetic)
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Add value labels on bars
def autolabel(bars, ax, fmt='{:.3f}'):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(fmt.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(bars1, ax1)
autolabel(bars2, ax1)
autolabel(bars3, ax1)

# Real Data (PeerJ) Results
approaches_real = ['Baseline\n(Fixed Weights)', 'Standard\nLOTO', 'Uncertainty-Guided\nTransfer Learning']
r2_real = [0.572, 0.572, 0.591]
ece_real = [0.083, 0.083, 0.078]
coverage_real = [0.917, 0.917, 0.925]

x = np.arange(len(approaches_real))
width = 0.25

bars1 = ax2.bar(x - width, r2_real, width, label='R²', color='skyblue', edgecolor='navy')
bars2 = ax2.bar(x, ece_real, width, label='ECE (lower better)', color='lightcoral', edgecolor='darkred')
bars3 = ax2.bar(x + width, coverage_real, width, label='90% Coverage', color='lightgreen', edgecolor='darkgreen')

ax2.set_xlabel('Approach', fontsize=12, fontweight='bold')
ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
ax2.set_title('Real PeerJ Dataset Performance\n(}t ranslational Validation)', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(approaches_real)
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

# Add value labels on bars
autolabel(bars1, ax2)
autolabel(bars2, ax2)
autolabel(bars3, ax2)

# Overall title
fig.suptitle('Performance Comparison: Quality-Aware Fusion with Uncertainty-Guided Transfer Learning',
             fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('C:\\Users\\priti\\OneDrive\\Desktop\\capstone\\research_paper\\results\\figures\\results_summary.png',
            dpi=300, bbox_inches='tight')
plt.close()

print("Results summary figure saved to: results/figures/results_summary.png")