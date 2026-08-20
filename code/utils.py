import numpy as np
import torch
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os

def create_windows(data, window_len=30, step=1, target_keys=None):
    """
    Create sliding windows for time series prediction.
    data: dict with keys as signal names, each a 1D numpy array.
    Returns:
        X: numpy array of shape (n_windows, window_len, num_features)
        y: dict of targets for each key in target_keys - shape (n_windows,)
    """
    # We'll predict the next second after the window
    # Signals to use as features: all keys in data
    feature_keys = list(data.keys())
    # If target_keys not provided, predict all features
    if target_keys is None:
        target_keys = list(data.keys())

    # Stack features
    feature_arrays = [data[key] for key in feature_keys]
    X = np.stack(feature_arrays, axis=-1)  # (T, num_features)

    # Targets: we will predict the value at t+1 (next second)
    # So for window ending at t, we want t+1
    # We'll create windows such that the last element of the window is at index t,
    # and the target is at index t+1.
    max_start = len(X) - window_len - 1  # need at least one step ahead
    if max_start < 0:
        raise ValueError("Not enough data for given window length.")

    X_win = []
    y_dict = {key: [] for key in target_keys}

    for start in range(0, max_start + 1, step):
        end = start + window_len  # exclusive
        X_win.append(X[start:end])  # (window_len, num_features)
        # Target is at index end (because we want t+1 where t = end-1)
        for key in target_keys:
            y_dict[key].append(data[key][end])

    X_win = np.stack(X_win, axis=0)  # (n_windows, window_len, num_features)
    for key in target_keys:
        y_dict[key] = np.array(y_dict[key])  # (n_windows,)

    return X_win, y_dict

def compute_r2(y_true, y_pred):
    """R^2 score."""
    return r2_score(y_true, y_pred)

def expected_calibration_error(y_true, y_mean, y_std, num_bins=10):
    """
    Compute Expected Calibration Error (ECE) for regression.
    We bin predictions by predicted standard deviation and compute
    the average absolute difference between empirical coverage and confidence.
    Reference: "On Calibration of Modern Neural Networks" (Guo et al., 2017) adapted for regression.
    Steps:
        1. For each prediction, compute standardized residual: z = (y_true - y_mean) / y_std
        2. Under perfect calibration, z should be ~ N(0,1). We check the proportion of |z| <= z_threshold
           for various confidence levels.
        3. Bin by predicted std (or by confidence) and compute calibration error.
    We'll follow a simple approach: bin by predicted std, compute empirical coverage of 90% interval.
    """
    # 90% interval corresponds to |z| <= 1.645 (approx)
    z_threshold = 1.645
    # Compute standardized residuals
    z = np.abs((y_true - y_mean) / y_std)
    inside_90 = z <= z_threshold

    # Bin by predicted std (y_std)
    # We'll create num_bins bins based on y_std values
    bins = np.linspace(np.min(y_std), np.max(y_std), num_bins + 1)
    bin_indices = np.digitize(y_std, bins) - 1  # 0..num_bins-1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    ece = 0.0
    for b in range(num_bins):
        mask = bin_indices == b
        if np.sum(mask) == 0:
            continue
        prop_inside = np.mean(inside_90[mask])
        # Expected proportion under perfect calibration for 90% interval is 0.9
        ece += np.sum(mask) / len(y_true) * np.abs(prop_inside - 0.9)
    return ece

def prediction_interval_coverage(y_true, y_mean, y_std, ci=0.9):
    """
    Compute fraction of true values inside the predicted confidence interval.
    Assuming Gaussian predictive distribution.
    """
    from scipy import stats
    z = stats.norm.ppf((1 + ci) / 2)
    lower = y_mean - z * y_std
    upper = y_mean + z * y_std
    inside = (y_true >= lower) & (y_true <= upper)
    return np.mean(inside)

def credible_interval_coverage(posterior_samples, y_true, ci=0.9):
    """
    Compute fraction of true values inside the central credible interval.
    posterior_samples: (num_samples, N_test)
    y_true: (N_test,)
    Returns coverage proportion.
    """
    lower = np.percentile(posterior_samples, (1 - ci) / 2 * 100, axis=0)
    upper = np.percentile(posterior_samples, (1 + ci) / 2 * 100, axis=0)
    inside = (y_true >= lower) & (y_true <= upper)
    return np.mean(inside)

def plot_uaft_accuracy(time_minutes, y_true, y_mean, y_std, save_path):
    """Plot UAFT predictions vs truth with uncertainty band."""
    plt.figure(figsize=(10, 4))
    plt.plot(time_minutes, y_true, label='True β‑HB (scaled)', color='black', linewidth=2)
    plt.plot(time_minutes, y_mean, label='UAFT mean', color='blue', linewidth=2)
    plt.fill_between(time_minutes,
                     y_mean - 1.645 * y_std,
                     y_mean + 1.645 * y_std,
                     color='blue', alpha=0.2, label='90% prediction interval')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Acetone (ppm) / β‑HB scaled')
    plt.title('UAFT: Prediction vs. Ground Truth')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_hbki_uncertainty(time_minutes, y_true, posterior_mean, posterior_samples, save_path):
    """Plot HBKI posterior mean and 90% credible interval."""
    lower = np.percentile(posterior_samples, 5, axis=0)
    upper = np.percentile(posterior_samples, 95, axis=0)
    plt.figure(figsize=(10, 4))
    plt.plot(time_minutes, y_true, label='True β‑HB (scaled)', color='black', linewidth=2)
    plt.plot(time_minutes, posterior_mean, label='HBKI mean', color='orange', linewidth=2)
    plt.fill_between(time_minutes, lower, upper, color='orange', alpha=0.3, label='90% credible interval')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Ketosis score (0‑100)')
    plt.title('HBKI: Posterior Estimate vs. Ground Truth')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_ablation(metrics_dict, save_path):
    """Create a bar chart for ablation study."""
    # metrics_dict: {scenario: {metric: value}}
    # We'll plot R² and ECE for each scenario.
    scenarios = list(metrics_dict.keys())
    r2_vals = [metrics_dict[s].get('r2', 0) for s in scenarios]
    ece_vals = [metrics_dict[s].get('ece', 0) for s in scenarios]

    x = np.arange(len(scenarios))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 4))
    bars1 = ax1.bar(x - width/2, r2_vals, width, label='R²', color='skyblue')
    ax1.set_ylabel('R²', color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, rotation=15, ha='right')

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, ece_vals, width, label='ECE', color='lightcoral')
    ax2.set_ylabel('Expected Calibration Error (ECE)', color='lightcoral')
    ax2.tick_params(axis='y', labelcolor='lightcoral')

    # Add value labels on bars
    def autolabel(bars, ax):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    autolabel(bars1, ax1)
    autolabel(bars2, ax2)

    fig.tight_layout()
    plt.title('Ablation Study: Impact of Uncertainty‑Aware Weighting and User Context')
    plt.savefig(save_path, dpi=150)
    plt.close()

def load_peerj_dataset(filepath):
    """
    Load the PeerJ breath-blood dataset (peerj-08-9969-s006.txt).
    Expected columns: UserID timestamp Blood (mM) Breath (ACEs) Breath (ppm)
    Returns a dictionary with arrays:
        user_id: subject IDs (int)
        timestamp: raw Unix timestamps
        time_hours: hours since first timestamp (float)
        blood_mM: blood biomarker concentration (mM)
        breath_aces: normalized breath measurement (ACEs)
        breath_ppm: breath concentration in parts per million
    """
    # Skip comment lines starting with '#'
    data = np.loadtxt(filepath, comments='#', delimiter=None)

    # Extract columns
    user_id = data[:, 0].astype(int)
    timestamp = data[:, 1]
    blood_mM = data[:, 2]
    breath_aces = data[:, 3]
    breath_ppm = data[:, 4]

    # Convert timestamp to hours since start for easier modeling
    time_hours = (timestamp - timestamp[0]) / 3600.0

    return {
        'user_id': user_id,
        'timestamp': timestamp,
        'time_hours': time_hours,
        'blood_mM': blood_mM,
        'breath_aces': breath_aces,
        'breath_ppm': breath_ppm
    }


if __name__ == "__main__":
    # Quick test of windowing (using dummy data)
    dummy_len = 100
    dummy_data = {
        'acetone_ppm': np.random.randn(dummy_len),
        'ammonia_ppm': np.random.randn(dummy_len),
        'methane_ppm': np.random.randn(dummy_len),
        'flow_Lpm': np.random.randn(dummy_len),
        'co2_pct': np.random.randn(dummy_len),
    }
    X, y_dict = create_windows(dummy_data, window_len=10, step=1)
    print(f"Windowed data shape: X={X.shape}, y acetone shape={y_dict['acetone_ppm'].shape}")