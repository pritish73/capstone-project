"""
Evaluation of the uncertainty-aware framework on real PeerJ breath-blood data.
Uses only breath ppm measurement and derived features to predict blood mM.
Demonstrates quality-aware learning and uncertainty quantification.
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from utils import load_peerj_dataset
from hbki import AdaptiveBayesianRidge, hbki_fit, hbki_predict, compute_metrics
from sklearn.metrics import r2_score
from sklearn.linear_model import BayesianRidge

def create_features_breath_only(breath_ppm, window_len=10):
    """
    Create features from breath_ppm signal:
        - raw breath_ppm
        - rolling mean (window_len)
        - rolling std (window_len) -> inverse quality indicator
        - first difference (breath_ppm[t] - breath_ppm[t-1])
    Returns feature matrix of shape (n_samples, n_features)
    """
    n = len(breath_ppm)
    # raw
    f0 = breath_ppm.copy()
    # rolling mean
    f1 = np.convolve(breath_ppm, np.ones(window_len)/window_len, mode='same')
    # rolling std
    # using convolution for std approximation: sqrt(conv(x^2) - mean^2)
    x_sq = breath_ppm ** 2
    mean_sq = np.convolve(x_sq, np.ones(window_len)/window_len, mode='same')
    mean = np.convolve(breath_ppm, np.ones(window_len)/window_len, mode='same')
    f2 = np.sqrt(np.maximum(mean_sq - mean**2, 0))
    # first difference
    f3 = np.diff(breath_ppm, prepend=breath_ppm[0])

    # Stack
    features = np.vstack([f0, f1, f2, f3]).T  # (n, 4)
    return features

def evaluate_subject_split(data, feature_fn, window_len=10):
    """
    Leave-one-subject-out cross-validation.
    For each subject, train on all other subjects, test on held-out subject.
    Returns aggregated predictions and truths.
    """
    user_ids = data['user_id']
    unique_ids = np.unique(user_ids)
    all_pred = []
    all_true = []
    all_std = []

    for test_id in unique_ids:
        train_mask = user_ids != test_id
        test_mask = user_ids == test_id

        # Training data
        breath_train = data['breath_ppm'][train_mask]
        blood_train = data['blood_mM'][train_mask]

        # Create features
        X_train = feature_fn(breath_train, window_len=window_len)
        # Standardize using training set
        mean_X = np.mean(X_train, axis=0)
        std_X = np.std(X_train, axis=0)
        std_X[std_X < 1e-8] = 1.0
        X_train_std = (X_train - mean_X) / std_X

        # Test data
        breath_test = data['breath_ppm'][test_mask]
        blood_test = data['blood_mM'][test_mask]
        X_test = feature_fn(breath_test, window_len=window_len)
        X_test_std = (X_test - mean_X) / std_X

        # Train model (Adaptive Bayesian Ridge)
        model = AdaptiveBayesianRidge(
            alpha_1=1e-6, alpha_2=1e-6,
            lambda_1=1e-6, lambda_2=1e-6,
            compute_score=True,
            fit_intercept=True
        )
        model.fit(X_train_std, blood_train)

        # Predict
        pred_mean, pred_std = model.predict(X_test_std, return_std=True)

        all_pred.append(pred_mean)
        all_true.append(blood_test)
        all_std.append(pred_std)

    # Concatenate
    pred_mean = np.concatenate(all_pred)
    true_vals = np.concatenate(all_true)
    pred_std = np.concatenate(all_std)

    return pred_mean, true_vals, pred_std

def evaluate_fixed_weights(data, feature_fn, window_len=10):
    """
    Baseline: standard Bayesian Ridge with fixed (uniform) weights.
    """
    user_ids = data['user_id']
    unique_ids = np.unique(user_ids)
    all_pred = []
    all_true = []
    all_std = []

    for test_id in unique_ids:
        train_mask = user_ids != test_id
        test_mask = user_ids == test_id

        breath_train = data['breath_ppm'][train_mask]
        blood_train = data['blood_mM'][train_mask]

        X_train = feature_fn(breath_train, window_len=window_len)
        mean_X = np.mean(X_train, axis=0)
        std_X = np.std(X_train, axis=0)
        std_X[std_X < 1e-8] = 1.0
        X_train_std = (X_train - mean_X) / std_X

        breath_test = data['breath_ppm'][test_mask]
        blood_test = data['blood_mM'][test_mask]
        X_test = feature_fn(breath_test, window_len=window_len)
        X_test_std = (X_test - mean_X) / std_X

        model = BayesianRidge(
            alpha_1=1e-6, alpha_2=1e-6,
            lambda_1=1e-6, lambda_2=1e-6,
            compute_score=True,
            fit_intercept=True
        )
        model.fit(X_train_std, blood_train)

        pred_mean, pred_std = model.predict(X_test_std, return_std=True)

        all_pred.append(pred_mean)
        all_true.append(blood_test)
        all_std.append(pred_std)

    pred_mean = np.concatenate(all_pred)
    true_vals = np.concatenate(all_true)
    pred_std = np.concatenate(all_std)

    return pred_mean, true_vals, pred_std

def main():
    print("Loading PeerJ real dataset...")
    data = load_peerj_dataset('data/peerj-08-9969-s006.txt')

    print(f"Loaded {len(data['user_id'])} samples from {np.unique(data['user_id']).size} subjects.")

    # Evaluate with adaptive feature weighting (our method)
    print("\n=== Evaluating Adaptive Feature Weighting (HBKI-like) ===")
    pred_adapt, true, std_adapt = evaluate_subject_split(data, create_features_breath_only, window_len=10)

    # Compute metrics
    r2_adapt = r2_score(true, pred_adapt)
    # Expected Calibration Error (using utils)
    from utils import expected_calibration_error
    ece_adapt = expected_calibration_error(true, pred_adapt, std_adapt)
    # 90% prediction interval coverage
    from utils import prediction_interval_coverage
    cov_adapt = prediction_interval_coverage(true, pred_adapt, std_adapt, ci=0.9)

    print(f"R²: {r2_adapt:.4f}")
    print(f"ECE: {ece_adapt:.4f}")
    print(f"90% Prediction Interval Coverage: {cov_adapt:.4f}")

    # Evaluate baseline: fixed weights (standard Bayesian Ridge)
    print("\n=== Evaluating Baseline (Fixed Weights, Bayesian Ridge) ===")
    pred_fixed, true2, std_fixed = evaluate_fixed_weights(data, create_features_breath_only, window_len=10)

    r2_fixed = r2_score(true2, pred_fixed)
    ece_fixed = expected_calibration_error(true2, pred_fixed, std_fixed)
    cov_fixed = prediction_interval_coverage(true2, pred_fixed, std_fixed, ci=0.9)

    print(f"R²: {r2_fixed:.4f}")
    print(f"ECE: {ece_fixed:.4f}")
    print(f"90% Prediction Interval Coverage: {cov_fixed:.4f}")

    # Print improvement
    print("\n=== Improvement over Baseline ===")
    print(f"Delta R2: {r2_adapt - r2_fixed:.4f}")
    print(f"Delta ECE: {ece_fixed - ece_adapt:.4f} (positive means improvement)")
    print(f"Delta Coverage: {cov_adapt - cov_fixed:.4f}")

    # Feature weights from last fold (for inspection)
    # We'll show weights from one model (not aggregated)
    # Quick retrain on full data to see overall weights
    X_full = create_features_breath_only(data['breath_ppm'], window_len=10)
    mean_X = np.mean(X_full, axis=0)
    std_X = np.std(X_full, axis=0)
    std_X[std_X < 1e-8] = 1.0
    X_full_std = (X_full - mean_X) / std_X

    model_full = AdaptiveBayesianRidge(
        alpha_1=1e-6, alpha_2=1e-6,
        lambda_1=1e-6, lambda_2=1e-6,
        compute_score=True,
        fit_intercept=True
    )
    model_full.fit(X_full_std, data['blood_mM'])
    weights = model_full.get_feature_weights()
    if weights is not None:
        feature_names = ['breath_ppm', 'rolling_mean', 'rolling_std', 'first_diff']
        print("\nLearned feature weights (Adaptive Bayesian Ridge on full data):")
        for name, w in zip(feature_names, weights):
            print(f"  {name}: {w:.4f}")

    # Save results to file for later integration
    with open('results/real_evaluation.txt', 'w') as f:
        f.write("=== Real Data Evaluation (PeerJ dataset) ===\n")
        f.write(f"Number of samples: {len(data['user_id'])}\n")
        f.write(f"Number of subjects: {np.unique(data['user_id']).size}\n\n")
        f.write("=== Adaptive Feature Weighting (HBKI-like) ===\n")
        f.write(f"R²: {r2_adapt:.4f}\n")
        f.write(f"ECE: {ece_adapt:.4f}\n")
        f.write(f"90% Prediction Interval Coverage: {cov_adapt:.4f}\n\n")
        f.write("=== Baseline (Fixed Weights) ===\n")
        f.write(f"R²: {r2_fixed:.4f}\n")
        f.write(f"ECE: {ece_fixed:.4f}\n")
        f.write(f"90% Prediction Interval Coverage: {cov_fixed:.4f}\n\n")
        f.write("=== Improvement ===\n")
        f.write(f"Delta R2: {r2_adapt - r2_fixed:.4f}\n")
        f.write(f"Delta ECE: {ece_fixed - ece_adapt:.4f}\n")
        f.write(f"Delta Coverage: {cov_adapt - cov_fixed:.4f}\n")
        if weights is not None:
            f.write("\nFeature weights:\n")
            for name, w in zip(feature_names, weights):
                f.write(f"  {name}: {w:.4f}\n")

    print("\nResults saved to results/real_evaluation.txt")

if __name__ == "__main__":
    main()