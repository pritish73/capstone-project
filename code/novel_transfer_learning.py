"""
Novel Methodological Contribution: Uncertainty-Guided Transfer Learning
for Reducing Inter-Subject Variability in Breath Analysis

Approach:
1. For each test subject, train a model on all OTHER subjects
2. For each test sample, compute uncertainty-weighted similarity to training subjects
3. Make predictions using uncertainty-weighted combination of subject-level models
4. This reduces inter-subject variability by adapting to subject-specific characteristics
   while leveraging population knowledge
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from utils import load_peerj_dataset
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import r2_score
from utils import expected_calibration_error, prediction_interval_coverage

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
    x_sq = breath_ppm ** 2
    mean_sq = np.convolve(x_sq, np.ones(window_len)/window_len, mode='same')
    mean = np.convolve(breath_ppm, np.ones(window_len)/window_len, mode='same')
    f2 = np.sqrt(np.maximum(mean_sq - mean**2, 0))
    # first difference
    f3 = np.diff(breath_ppm, prepend=breath_ppm[0])

    # Stack
    features = np.vstack([f0, f1, f2, f3]).T  # (n, 4)
    return features

def uncertainty_weighted_transfer_learning(data, window_len=10):
    """
    Uncertainty-guided transfer learning approach:
    1. For each test subject, create features separately per subject to avoid leakage
    2. For each test subject, train a separate model on each training subject's data
    3. For each test sample, compute uncertainty for each TRAINING SUBJECT'S MODEL
    4. Use uncertainty to weight training subjects by similarity and reliability
    5. Predict using uncertainty-weighted combination of subject-level predictions
    """
    user_ids = data['user_id']
    unique_ids = np.unique(user_ids)
    n_subjects = len(unique_ids)

    # Create features per subject
    all_features = []
    all_targets = []
    all_user_ids = []

    for subject_id in unique_ids:
        subject_mask = user_ids == subject_id
        breath_subject = data['breath_ppm'][subject_mask]
        blood_subject = data['blood_mM'][subject_mask]

        if len(breath_subject) < window_len:
            continue

        features_subject = create_features_breath_only(breath_subject, window_len=window_len)
        all_features.append(features_subject)
        all_targets.append(blood_subject)
        all_user_ids.append(np.full(len(breath_subject), subject_id))

    if not all_features:
        # Fallback: create features on full data if no subject has enough data
        features_full = create_features_breath_only(data['breath_ppm'], window_len=window_len)
        return features_full, data['blood_mM'], data['user_id']

    features_all = np.vstack(all_features)
    targets_all = np.concatenate(all_targets)
    user_ids_all = np.concatenate(all_user_ids)

    all_pred = []
    all_true = []
    all_std = []

    print(f"Processing {n_subjects} subjects...")

    for test_idx, test_id in enumerate(unique_ids):
        if test_idx % 5 == 0:
            print(f"  Processing subject {test_id} ({test_idx+1}/{n_subjects})")

        train_mask = user_ids_all != test_id
        test_mask = user_ids_all == test_id

        features_train = features_all[train_mask]
        targets_train = targets_all[train_mask]
        features_test = features_all[test_mask]
        targets_test = targets_all[test_mask]

        print(f"  Train samples: {len(features_train)}, Test samples: {len(features_test)}")

        if len(features_train) == 0:
            print("  No training data, skipping")
            continue

        # Standardize using training set (for similarity and maybe fallback)
        mean_X = np.mean(features_train, axis=0)
        std_X = np.std(features_train, axis=0)
        std_X[std_X < 1e-8] = 1.0
        features_train_std = (features_train - mean_X) / std_X
        features_test_std_global = (features_test - mean_X) / std_X  # global std

        # Train separate model for each training subject
        unique_train_ids = np.unique(user_ids_all[train_mask])
        print(f"  Number of training subjects: {len(unique_train_ids)}")

        subject_models = {}
        for train_id in unique_train_ids:
            train_subject_mask = user_ids_all[train_mask] == train_id
            if np.sum(train_subject_mask) < 5:
                print(f"    Subject {train_id}: too few samples ({np.sum(train_subject_mask)}), skipping")
                continue

            features_subject = features_train[train_subject_mask]
            targets_subject = targets_train[train_subject_mask]

            # Standardize using this subject's data
            mean_subj = np.mean(features_subject, axis=0)
            std_subj = np.std(features_subject, axis=0)
            std_subj[std_subj < 1e-8] = 1.0
            features_subject_std = (features_subject - mean_subj) / std_subj

            subject_model = BayesianRidge(
                alpha_1=1e-6, alpha_2=1e-6,
                lambda_1=1e-6, lambda_2=1e-6,
                compute_score=True,
                fit_intercept=True
            )
            subject_model.fit(features_subject_std, targets_subject)

            subject_models[train_id] = {
                'model': subject_model,
                'mean': mean_subj,
                'std': std_subj
            }

        if not subject_models:
            print("  No valid subject models, using overall model")
            # Fallback: train overall model
            overall_model = BayesianRidge(
                alpha_1=1e-6, alpha_2=1e-6,
                lambda_1=1e-6, lambda_2=1e-6,
                compute_score=True,
                fit_intercept=True
            )
            overall_model.fit(features_train_std, targets_train)
            pred_mean, pred_std = overall_model.predict(features_test_std_global, return_std=True)
            all_pred.append(pred_mean)
            all_true.append(targets_test)
            all_std.append(pred_std)
            continue

        # For each test sample, compute uncertainty-weighted prediction
        test_predictions = []
        test_uncertainties = []

        for i in range(len(features_test)):
            # Get raw test feature vector
            test_sample_raw = features_test[i:i+1]  # shape (1, n_features)
            test_target = targets_test[i]

            subject_weights = []
            subject_preds = []
            subject_stds = []

            for train_id, model_info in subject_models.items():
                subject_model = model_info['model']
                subj_mean, subj_std = model_info['mean'], model_info['std']

                # Standardize test sample using THIS subject's mean and std
                test_sample_std = (test_sample_raw - subj_mean) / subj_std

                # Predict using subject model
                pred_mean_subj, pred_std_subj = subject_model.predict(test_sample_std, return_std=True)
                pred_mean_subj = pred_mean_subj[0]
                pred_std_subj = pred_std_subj[0]

                # Similarity to this subject's training data
                features_subject_train = features_train[user_ids_all[train_mask] == train_id]
                if len(features_subject_train) > 0:
                    # Use subject's own mean/std for standardization
                    mean_subj_train = model_info['mean']
                    std_subj_train = model_info['std']
                    features_subject_train_std = (features_subject_train - mean_subj_train) / std_subj_train

                    distances = np.sqrt(np.sum((features_subject_train_std - test_sample_std)**2, axis=1))
                    avg_distance = np.mean(distances) if len(distances) > 0 else 0.0

                    if len(distances) > 1:
                        bandwidth = np.median(distances)
                        if bandwidth < 1e-8:
                            bandwidth = 1.0
                        similarity = np.exp(-avg_distance**2 / (2 * bandwidth**2))
                    else:
                        similarity = 1.0
                else:
                    similarity = 0.0

                reliability = 1.0 / (pred_std_subj + 1e-8)
                combined_weight = similarity * reliability

                subject_weights.append(combined_weight)
                subject_preds.append(pred_mean_subj)
                subject_stds.append(pred_std_subj)

            if np.sum(subject_weights) > 0:
                subject_weights = np.array(subject_weights)
                subject_weights = subject_weights / np.sum(subject_weights)
                weighted_pred = np.sum(subject_weights * np.array(subject_preds))
                pred_variance = np.sum(subject_weights * np.array(subject_stds)**2)
                mean_pred = np.sum(subject_weights * np.array(subject_preds))
                between_variance = np.sum(subject_weights * (np.array(subject_preds) - mean_pred)**2)
                weighted_uncertainty = np.sqrt(pred_variance + between_variance)
            else:
                weighted_pred = np.mean(targets_train)
                weighted_uncertainty = np.std(targets_train) + 1e-8

            test_predictions.append(weighted_pred)
            test_uncertainties.append(weighted_uncertainty)

        all_pred.append(np.array(test_predictions))
        all_true.append(targets_test)
        all_std.append(np.array(test_uncertainties))

        if len(all_pred) > 0:
            partial_pred = np.concatenate(all_pred)
            partial_true = np.concatenate(all_true)
            partial_r2 = r2_score(partial_true, partial_pred)
            print(f"  Partial R² so far: {partial_r2:.4f}")

    # Final results
    pred_mean = np.concatenate(all_pred)
    true_vals = np.concatenate(all_true)
    pred_std = np.concatenate(all_std)

    print(f"\nFinal predictions shape: {pred_mean.shape}")
    print(f"Final true values shape: {true_vals.shape}")
    print(f"Final uncertainties shape: {pred_std.shape}")

    r2 = r2_score(true_vals, pred_mean)
    print(f"Final R²: {r2:.4f}")

    # Compute Expected Calibration Error (ECE) and coverage
    ece = expected_calibration_error(true_vals, pred_mean, pred_std)
    cov = prediction_interval_coverage(true_vals, pred_mean, pred_std, ci=0.9)
    print(f"ECE: {ece:.4f}")
    print(f"90% Prediction Interval Coverage: {cov:.4f}")

    return pred_mean, true_vals, pred_std

def standard_leave_one_out(data, window_len=10):
    """
    Standard leave-one-subject-out for comparison
    """
    # Create features separately per subject to avoid leakage
    user_ids = data['user_id']
    unique_ids = np.unique(user_ids)

    # Create features per subject
    all_features = []
    all_targets = []
    all_user_ids = []

    for subject_id in unique_ids:
        subject_mask = user_ids == subject_id
        breath_subject = data['breath_ppm'][subject_mask]
        blood_subject = data['blood_mM'][subject_mask]

        if len(breath_subject) < window_len:
            continue

        features_subject = create_features_breath_only(breath_subject, window_len=window_len)
        all_features.append(features_subject)
        all_targets.append(blood_subject)
        all_user_ids.append(np.full(len(breath_subject), subject_id))

    if not all_features:
        # Fallback: create features on full data if no subject has enough data
        features_full = create_features_breath_only(data['breath_ppm'], window_len=window_len)
        return features_full, data['blood_mM'], data['user_id']

    features_all = np.vstack(all_features)
    targets_all = np.concatenate(all_targets)
    user_ids_all = np.concatenate(all_user_ids)

    user_ids_array = user_ids_all
    unique_ids_array = np.unique(user_ids_array)

    all_pred = []
    all_true = []
    all_std = []

    for test_id in unique_ids_array:
        train_mask = user_ids_array != test_id
        test_mask = user_ids_array == test_id

        features_train = features_all[train_mask]
        targets_train = targets_all[train_mask]
        features_test = features_all[test_mask]
        targets_test = targets_all[test_mask]

        # Skip if insufficient training data
        if len(features_train) == 0:
            continue

        # Standardize using training set statistics
        mean_X = np.mean(features_train, axis=0)
        std_X = np.std(features_train, axis=0)
        std_X[std_X < 1e-8] = 1.0
        features_train_std = (features_train - mean_X) / std_X
        features_test_std = (features_test - mean_X) / std_X

        model = BayesianRidge(
            alpha_1=1e-6, alpha_2=1e-6,
            lambda_1=1e-6, lambda_2=1e-6,
            compute_score=True,
            fit_intercept=True
        )
        model.fit(features_train_std, targets_train)

        pred_mean, pred_std = model.predict(features_test_std, return_std=True)

        all_pred.append(pred_mean)
        all_true.append(targets_test)
        all_std.append(pred_std)

    pred_mean = np.concatenate(all_pred)
    true_vals = np.concatenate(all_true)
    pred_std = np.concatenate(all_std)

    return pred_mean, true_vals, pred_std

def main():
    print("Loading PeerJ real dataset...")
    data = load_peerj_dataset('data/peerj-08-9969-s006.txt')

    print(f"Loaded {len(data['user_id'])} samples from {np.unique(data['user_id']).size} subjects.")

    # Evaluate novel uncertainty-guided transfer learning
    print("\n=== Novel Approach: Uncertainty-Guided Transfer Learning ===")
    pred_novel, true_novel, std_novel = uncertainty_weighted_transfer_learning(data, window_len=10)

    r2_novel = r2_score(true_novel, pred_novel)
    ece_novel = expected_calibration_error(true_novel, pred_novel, std_novel)
    cov_novel = prediction_interval_coverage(true_novel, pred_novel, std_novel, ci=0.9)

    print(f"R²: {r2_novel:.4f}")
    print(f"ECE: {ece_novel:.4f}")
    print(f"90% Prediction Interval Coverage: {cov_novel:.4f}")

    # Evaluate standard leave-one-out for comparison
    print("\n=== Comparison: Standard Leave-One-Subject-Out ===")
    pred_standard, true_standard, std_standard = standard_leave_one_out(data, window_len=10)

    r2_standard = r2_score(true_standard, pred_standard)
    ece_standard = expected_calibration_error(true_standard, pred_standard, std_standard)
    cov_standard = prediction_interval_coverage(true_standard, pred_standard, std_standard, ci=0.9)

    print(f"R²: {r2_standard:.4f}")
    print(f"ECE: {ece_standard:.4f}")
    print(f"90% Prediction Interval Coverage: {cov_standard:.4f}")

    # Print improvement
    print("\n=== Improvement over Standard LOTO ===")
    print(f"Delta R²: {r2_novel - r2_standard:.4f}")
    print(f"Delta ECE: {ece_standard - ece_novel:.4f} (positive means improvement)")
    print(f"Delta Coverage: {cov_novel - cov_standard:.4f}")

    # Save results
    with open('results/novel_transfer_learning.txt', 'w') as f:
        f.write("=== Novel Methodological Contribution ===\n")
        f.write("Uncertainty-Guided Transfer Learning for Breath Analysis\n\n")
        f.write(f"Number of samples: {len(data['user_id'])}\n")
        f.write(f"Number of subjects: {np.unique(data['user_id']).size}\n\n")
        f.write("=== Novel Approach: Uncertainty-Guided Transfer Learning ===\n")
        f.write(f"R²: {r2_novel:.4f}\n")
        f.write(f"ECE: {ece_novel:.4f}\n")
        f.write(f"90% Prediction Interval Coverage: {cov_novel:.4f}\n\n")
        f.write("=== Standard Leave-One-Subject-Out (Baseline) ===\n")
        f.write(f"R²: {r2_standard:.4f}\n")
        f.write(f"ECE: {ece_standard:.4f}\n")
        f.write(f"90% Prediction Interval Coverage: {cov_standard:.4f}\n\n")
        f.write("=== Improvement (Novel - Standard) ===\n")
        f.write(f"Delta R²: {r2_novel - r2_standard:.4f}\n")
        f.write(f"Delta ECE: {ece_standard - ece_novel:.4f}\n")
        f.write(f"Delta Coverage: {cov_novel - cov_standard:.4f}\n\n")
        f.write("=== Approach Description ===\n")
        f.write("1. For each test subject, created separate features per subject to avoid leakage\n")
        f.write("2. Trained a separate Bayesian Ridge model on each training subject's data\n")
        f.write("3. For each test sample, computed similarity to each training subject's feature space\n")
        f.write("4. Weighted predictions by similarity and reliability (inverse uncertainty)\n")
        f.write("5. Made final prediction as uncertainty-weighted combination of subject-level predictions\n")
        f.write("6. This reduces inter-subject variability by adapting to subject-specific characteristics\n")
        f.write("   while leveraging population knowledge through uncertainty-guided weighting\n")

    print("\nResults saved to results/novel_transfer_learning.txt")

if __name__ == "__main__":
    main()