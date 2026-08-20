import sys
sys.path.insert(0, '.')
from utils import load_peerj_dataset
import numpy as np

# Load dataset
data = load_peerj_dataset('data/peerj-08-9969-s006.txt')

print("Dataset shape:")
print(f"  Number of samples: {len(data['user_id'])}")
print(f"  Unique subjects: {np.unique(data['user_id']).size}")
print(f"  Subject IDs: {np.unique(data['user_id'])}")

print("\nStatistics:")
print(f"  Blood mM: mean={np.mean(data['blood_mM']):.3f}, std={np.std(data['blood_mM']):.3f}, min={np.min(data['blood_mM']):.3f}, max={np.max(data['blood_mM']):.3f}")
print(f"  Breath ppm: mean={np.mean(data['breath_ppm']):.3f}, std={np.std(data['breath_ppm']):.3f}, min={np.min(data['breath_ppm']):.3f}, max={np.max(data['breath_ppm']):.3f}")
print(f"  Breath ACEs: mean={np.mean(data['breath_aces']):.3f}, std={np.std(data['breath_aces']):.3f}")

print("\nCorrelation between breath ppm and blood mM:")
corr = np.corrcoef(data['breath_ppm'], data['blood_mM'])[0,1]
print(f"  Pearson r = {corr:.4f}")

# Simple linear regression slope and intercept
A = np.vstack([data['breath_ppm'], np.ones(len(data['breath_ppm']))]).T
slope, intercept = np.linalg.lstsq(A, data['blood_mM'], rcond=None)[0]
print(f"  Linear regression: blood = {slope:.4f} * breath + {intercept:.4f}")

# Compute residuals and std of residuals
pred = slope * data['breath_ppm'] + intercept
residuals = data['blood_mM'] - pred
res_std = np.std(residuals)
print(f"  Residual std = {res_std:.4f}")

# Compute R^2
ss_res = np.sum(residuals**2)
ss_tot = np.sum((data['blood_mM'] - np.mean(data['blood_mM']))**2)
r2 = 1 - ss_res/ss_tot
print(f"  R^2 = {r2:.4f}")

# Check if there are temporal patterns per subject
print("\nPer-subject stats (first 5 subjects):")
for subj in np.unique(data['user_id'])[:5]:
    mask = data['user_id'] == subj
    print(f"  Subject {subj}: n={np.sum(mask)}, blood mean={np.mean(data['blood_mM'][mask]):.3f}, breath mean={np.mean(data['breath_ppm'][mask]):.3f}")