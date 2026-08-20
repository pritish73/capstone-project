import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, RegressorMixin
import warnings

class AdaptiveBayesianRidge(BaseEstimator, RegressorMixin):
    """
    Adaptive Bayesian Ridge regression with feature weighting that learns
    the importance of different quality indicators and gas features.
    """
    def __init__(self, alpha_1=1e-6, alpha_2=1e-6,
                 lambda_1=1e-6, lambda_2=1e-6,
                 compute_score=True, fit_intercept=True,
                 n_iter=300, tol=1e-3, random_state=None):
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.compute_score = compute_score
        self.fit_intercept = fit_intercept
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.feature_weights_ = None
        self.base_model_ = None
        self.iteration_history_ = []

    def fit(self, X, y):
        """
        Fit the adaptive Bayesian Ridge model.
        Learns feature weights based on prediction performance.
        """
        n_features = X.shape[1]

        # Initialize feature weights uniformly
        if self.feature_weights_ is None:
            self.feature_weights_ = np.ones(n_features) / n_features

        # Store initial weights
        self.iteration_history_.append(self.feature_weights_.copy())

        # Iteratively reweight features based on residuals
        for iteration in range(self.n_iter):
            # Apply current feature weights
            X_weighted = X * self.feature_weights_

            # Fit base model
            self.base_model_ = BayesianRidge(
                alpha_1=self.alpha_1, alpha_2=self.alpha_2,
                lambda_1=self.lambda_1, lambda_2=self.lambda_2,
                compute_score=self.compute_score,
                fit_intercept=self.fit_intercept,
                tol=self.tol
            )
            self.base_model_.fit(X_weighted, y)

            # Update feature weights based on absolute coefficients
            # Features with higher predictive power get higher weights
            coef_abs = np.abs(self.base_model_.coef_)
            if np.sum(coef_abs) > 0:
                new_weights = coef_abs / np.sum(coef_abs)
            else:
                # Fallback to uniform if all coefficients are zero
                new_weights = np.ones(n_features) / n_features

            # Check for convergence
            weight_change = np.sum(np.abs(new_weights - self.feature_weights_))
            self.feature_weights_ = new_weights
            self.iteration_history_.append(self.feature_weights_.copy())

            if weight_change < self.tol:
                break

        # Final fit with converged weights
        X_weighted = X * self.feature_weights_
        self.base_model_.fit(X_weighted, y)

        return self

    def predict(self, X, return_std=False):
        """
        Predict using the adaptive Bayesian Ridge model.
        """
        if self.base_model_ is None:
            raise ValueError("Model must be fitted before prediction")

        X_weighted = X * self.feature_weights_
        return self.base_model_.predict(X_weighted, return_std=return_std)

    def get_feature_weights(self):
        """
        Get the learned feature weights.
        """
        return self.feature_weights_.copy() if self.feature_weights_ is not None else None

    def get_iteration_history(self):
        """
        Get the history of feature weights across iterations.
        """
        return np.array(self.iteration_history_) if self.iteration_history_ else None

def hbki_fit(X_train, y_train):
    """
    Fit an Adaptive Bayesian Ridge regression model.
    Returns the fitted model.
    """
    model = AdaptiveBayesianRidge(
        alpha_1=1e-6, alpha_2=1e-6,
        lambda_1=1e-6, lambda_2=1e-6,
        compute_score=True,  # if we want to access log marginal likelihood
        fit_intercept=True
    )
    model.fit(X_train, y_train)
    return model

def hbki_predict(model, X_test):
    """
    Predict mean and variance using Adaptive Bayesian Ridge.
    Returns:
        mean: shape (n_samples,)
        std: shape (n_samples,)  (sqrt of variance)
    """
    mean, std = model.predict(X_test, return_std=True)
    return mean, std

def compute_metrics(y_true, y_pred, y_std=None):
    """Compute R^2 and optionally calibration metrics if y_std provided."""
    r2 = r2_score(y_true, y_pred)
    metrics = {'r2': r2}
    if y_std is not None:
        # Expected Calibration Error (ECE) for 90% interval
        from utils import expected_calibration_error, prediction_interval_coverage
        ece = expected_calibration_error(y_true, y_pred, y_std)
        cov_90 = prediction_interval_coverage(y_true, y_pred, y_std, ci=0.9)
        metrics['ece'] = ece
        metrics['coverage_90'] = cov_90

        # For BayesianRidge, we can approximate credible interval coverage
        # using the predictive std (assuming Gaussian)
        from utils import credible_interval_coverage
        from scipy import stats
        z = stats.norm.ppf((1 + 0.9) / 2)  # 90% interval
        lower = y_pred - z * y_std
        upper = y_pred + z * y_std
        credible_cov = credible_interval_coverage_from_samples(y_true, lower, upper)
        metrics['credible_coverage_90'] = credible_cov

    return metrics

def credible_interval_coverage_from_samples(y_true, lower, upper):
    """Compute fraction of true values inside [lower, upper]."""
    inside = (y_true >= lower) & (y_true <= upper)
    return np.mean(inside)

def predict_hbki(X_train, y_train, X_test, y_test=None):
    """
    Convenience function: fit BayesianRidge and predict on X_test.
    If y_test provided, also compute metrics.
    Returns:
        model: fitted BayesianRidge
        y_pred_mean: predicted mean
        y_pred_std: predicted std (if return_std=True in predict)
        metrics: dict if y_test provided, else None
    """
    model = hbki_fit(X_train, y_train)
    y_pred_mean, y_pred_std = hbki_predict(model, X_test)
    metrics = None
    if y_test is not None:
        metrics = compute_metrics(y_test, y_pred_mean, y_pred_std)
    return model, y_pred_mean, y_pred_std, metrics

def shap_guided_feature_selection(model, X, y, feature_names=None, threshold=0.01):
    """
    Use SHAP values to guide feature selection by removing low-impact features.
    Returns indices of features to keep.
    """
    try:
        import shap
        # Handle AdaptiveBayesianRidge model
        if hasattr(model, 'base_model_') and hasattr(model.base_model_, 'coef_'):
            # For linear models, SHAP values can be approximated as feature * coefficient
            # We'll use the absolute product as importance measure
            coef = model.base_model_.coef_

            # Compute mean absolute contribution per feature
            # SHAP value for linear model: phi_j = x_j * w_j
            # Mean |SHAP| = mean(|x_j * w_j|) over samples
            abs_contributions = np.abs(X * coef[np.newaxis, :])
            mean_abs_contrib = np.mean(abs_contributions, axis=0)

            # Normalize to get importance scores
            if np.sum(mean_abs_contrib) > 0:
                importance_scores = mean_abs_contrib / np.sum(mean_abs_contrib)
            else:
                importance_scores = np.ones_like(mean_abs_contrib) / len(mean_abs_contrib)

            # Select features above threshold
            selected_indices = np.where(importance_scores >= threshold)[0]

            # Ensure we keep at least one feature
            if len(selected_indices) == 0:
                selected_indices = [np.argmax(importance_scores)]

            return selected_indices
        else:
            # Fallback: use all features
            if feature_names is None:
                return np.arange(X.shape[1])
            else:
                return np.arange(len(feature_names))
    except ImportError:
        warnings.warn("SHAP not available, skipping feature selection")
        if feature_names is None:
            return np.arange(X.shape[1])
        else:
            return np.arange(len(feature_names))

def create_deterministic_baseline(X_train, y_train):
    """
    Create a deterministic baseline model (no uncertainty) for ablation studies.
    Uses standard linear regression with no uncertainty estimation.
    """
    from sklearn.linear_model import Ridge
    # Standard Ridge regression (deterministic, no uncertainty)
    model = Ridge(alpha=1e-6, fit_intercept=True, random_state=42)
    model.fit(X_train, y_train)
    return model

def predict_deterministic(model, X_test):
    """
    Predict using deterministic model (returns mean only).
    """
    mean = model.predict(X_test)
    # Return std as zeros or small constant to indicate no uncertainty
    std = np.zeros_like(mean)
    return mean, std