import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class UncertaintyAwareTransformer(nn.Module):
    """
    An ensemble of Transformer encoders that predicts mean and log-variance for each gas.
    Uses heteroscedastic uncertainty quantification with optional ensemble for improved calibration.
    Input shape: (batch, seq_len, num_features)
    Output: dict with 'mean' and 'logvar' for each gas (acetone, ammonia, methane, h2, h2s).
    Flow and CO2 are used as auxiliary features but not predicted.
    """
    def __init__(self, input_dim=6, num_gas=5, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, dropout=0.1, ensemble_size=3):
        super().__init__()
        self.input_dim = input_dim
        self.num_gas = num_gas
        self.d_model = d_model
        self.ensemble_size = ensemble_size
        self.dropout = dropout

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # Create ensemble of Transformer encoders
        self.ensemble = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                         dim_feedforward=dim_feedforward,
                                         dropout=dropout, activation='gelu'),
                num_layers=num_layers
            )
            for _ in range(ensemble_size)
        ])

        # Output heads: one for mean, one for log-variance, per gas
        self.mean_head = nn.Linear(d_model, num_gas)
        self.logvar_head = nn.Linear(d_model, num_gas)
        # Initialize output layers with small weights to prevent large outputs
        nn.init.xavier_uniform_(self.mean_head.weight)
        nn.init.xavier_uniform_(self.logvar_head.weight)
        nn.init.constant_(self.mean_head.bias, 0.0)
        nn.init.constant_(self.logvar_head.bias, -5.0)  # Initialize to predict small variance

    def forward(self, x, return_ensemble=False):
        """
        x: Tensor of shape (batch, seq_len, input_dim)
        Returns:
            mean: Tensor (batch, num_gas)
            logvar: Tensor (batch, num_gas) (clamped for stability)
            If return_ensemble=True, also returns list of individual ensemble member predictions
        """
        # Project input
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        # Transformer expects (seq_len, batch, d_model)
        x = x.permute(1, 0, 2)

        # Get predictions from each ensemble member
        ensemble_means = []
        ensemble_logvars = []

        for i in range(self.ensemble_size):
            # Pass through transformer encoder
            x_encoded = self.ensemble[i](x)  # (seq_len, batch, d_model)
            # We take the output of the last time step for prediction
            x_last = x_encoded[-1]  # (batch, d_model)

            mean = self.mean_head(x_last)      # (batch, num_gas)
            logvar = self.logvar_head(x_last)  # (batch, num_gas)
            # Clamp logvar to prevent extreme values
            logvar = torch.clamp(logvar, min=-10.0, max=10.0)

            ensemble_means.append(mean)
            ensemble_logvars.append(logvar)

        # Stack ensemble predictions
        ensemble_means = torch.stack(ensemble_means, dim=0)   # (ensemble_size, batch, num_gas)
        ensemble_logvars = torch.stack(ensemble_logvars, dim=0)  # (ensemble_size, batch, num_gas)

        # Compute mean and variance across ensemble
        mean_of_means = torch.mean(ensemble_means, dim=0)
        # Variance of means (epistemic uncertainty)
        var_of_means = torch.var(ensemble_means, dim=0)
        # Average of predicted variances (aleatoric uncertainty)
        # Convert logvar to var: var = exp(logvar)
        vars_from_logvar = torch.exp(ensemble_logvars)
        mean_vars = torch.mean(vars_from_logvar, dim=0)
        # Total variance = aleatoric + epistemic
        total_var = mean_vars + var_of_means
        # Convert back to logvar for consistency
        total_logvar = torch.log(total_var + 1e-8)

        if return_ensemble:
            return mean_of_means, total_logvar, ensemble_means, ensemble_logvars
        else:
            return mean_of_means, total_logvar

    def predict_with_uncertainty(self, x, num_samples=100):
        """
        Monte Carlo dropout for uncertainty estimation (alternative to ensemble).
        Returns mean and std of predictions across stochastic forward passes.
        """
        self.train()  # Enable dropout
        means = []
        logvars = []

        with torch.no_grad():
            for _ in range(num_samples):
                # Use first ensemble member for MC dropout (or average)
                mean, logvar, _, _ = self.forward(x, return_ensemble=True)
                means.append(mean)
                logvars.append(logvar)

        means = torch.stack(means)  # (num_samples, batch, num_gas)
        logvars = torch.stack(logvars)  # (num_samples, batch, num_gas)

        # Compute mean of means
        mean_of_means = torch.mean(means, dim=0)
        # Compute total uncertainty: aleatoric + epistemic
        # Aleatoric: average of predicted variances
        aleatoric = torch.mean(torch.exp(logvars), dim=0)
        # Epistemic: variance of means
        epistemic = torch.var(means, dim=0)
        # Total variance
        total_var = aleatoric + epistemic
        # Convert to logvar for consistency
        total_logvar = torch.log(total_var + 1e-8)

        return mean_of_means, total_logvar

def heteroscedastic_loss(y_true, y_mean, y_logvar):
    """
    Heteroscedastic Gaussian loss: log(var) + (y - mu)^2 / var
    where var = exp(logvar)
    y_true, y_mean, y_logvar: Tensors of same shape (batch, num_gas) or (batch,) if single gas.
    Returns scalar loss (mean over batch and gases).
    """
    # Ensure same shape
    if y_true.dim() == 1:
        y_true = y_true.unsqueeze(-1)
    if y_mean.dim() == 1:
        y_mean = y_mean.unsqueeze(-1)
    if y_logvar.dim() == 1:
        y_logvar = y_logvar.unsqueeze(-1)

    var = torch.exp(y_logvar)
    loss = y_logvar + (y_true - y_mean)**2 / var
    return loss.mean()

def standardize_targets(y_train, y_val=None, y_test=None):
    """
    Standardize targets to zero mean, unit variance using training set statistics.
    Returns standardized versions and the mean/std used for standardization.
    """
    y_mean = np.mean(y_train, axis=0)
    y_std = np.std(y_train, axis=0)
    # Avoid division by zero
    y_std[y_std < 1e-6] = 1.0

    y_train_std = (y_train - y_mean) / y_std
    y_val_std = (y_val - y_mean) / y_std if y_val is not None else None
    y_test_std = (y_test - y_mean) / y_std if y_test is not None else None

    return y_train_std, y_val_std, y_test_std, y_mean, y_std

def standardize_features(X_train, X_val=None, X_test=None):
    """
    Standardize features to zero mean, unit variance using training set statistics.
    Returns standardized versions and the mean/std used for standardization.
    """
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    # Avoid division by zero
    X_std[X_std < 1e-6] = 1.0

    X_train_std = (X_train - X_mean) / X_std
    X_val_std = (X_val - X_mean) / X_std if X_val is not None else None
    X_test_std = (X_test - X_mean) / X_std if X_test is not None else None

    return X_train_std, X_val_std, X_test_std, X_mean, X_std

def compute_uncertainty_calibration(y_true, y_mean, y_logvar, num_bins=15):
    """
    Compute uncertainty calibration metrics and return reliability diagram data.
    Adapted from classification uncertainty calibration to regression.
    """
    # Convert to numpy for easier binning
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_mean):
        y_mean = y_mean.detach().cpu().numpy()
    if torch.is_tensor(y_logvar):
        y_logvar = y_logvar.detach().cpu().numpy()

    var = np.exp(y_logvar)
    std = np.sqrt(var)

    # Compute standardized residuals: z = (y_true - y_mean) / std
    # Avoid division by zero
    std_safe = np.copy(std)
    std_safe[std_safe == 0] = 1e-8
    z_scores = np.abs((y_true - y_mean) / std_safe)

    # For well-calibrated uncertainties, z_scores should follow half-normal distribution
    # We'll check calibration at different confidence levels
    try:
        from scipy import stats
        confidence_levels = np.linspace(0.1, 0.9, 9)  # 0.1, 0.2, ..., 0.9
        z_thresholds = stats.norm.ppf((1 + confidence_levels) / 2)  # Two-tailed
    except ImportError:
        # Fallback approximation
        confidence_levels = np.linspace(0.1, 0.9, 9)
        z_thresholds = np.array([1.28, 1.64, 1.96, 2.33, 2.58, 2.81, 3.00, 3.20, 3.36])  # Approximate

    empirical_probs = []
    for i, threshold in enumerate(z_thresholds):
        # Fraction of points where |z| <= threshold (should equal confidence_levels[i])
        empirical_prob = np.mean(z_scores <= threshold)
        empirical_probs.append(empirical_prob)

    # Expected Calibration Error (ECE) for regression
    ece = np.mean(np.abs(np.array(empirical_probs) - confidence_levels))

    # Also compute negative log likelihood
    nll = 0.5 * np.log(2 * np.pi * var) + 0.5 * (y_true - y_mean)**2 / var
    mean_nll = np.mean(nll)

    return {
        'ece': ece,
        'mean_nll': mean_nll,
        'confidence_levels': confidence_levels,
        'empirical_probs': empirical_probs,
        'z_scores_mean': np.mean(z_scores),
        'z_scores_std': np.std(z_scores)
    }