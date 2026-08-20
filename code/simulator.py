import numpy as np
import os

def generate_breath_simulation(duration_hours=2, sample_rate_hz=1, seed=42):
    """
    Generate synthetic breath data simulating end-tidal breath analysis
    with physiologically grounded parameters from sensor datasheets.
    Includes: H2, CH4, H2S, NH3, acetone, plus environmental variables.
    Returns a dictionary with time series and ground truth physiological states.
    """
    np.random.seed(seed)

    # Time array
    t_span = (0, duration_hours * 3600)  # Convert hours to seconds
    t_eval = np.linspace(0, duration_hours * 3600, int(duration_hours * 3600 * sample_rate_hz))
    n_points = len(t_eval)

    # Initial conditions for physiological states
    # [acetone_blood (mM), acetoacetate_blood (mM), betaHB_blood (mM),
    #  H2_gut (ppm), CH4_gut (ppm), H2S_gut (ppm), NH3_blood (mM)]
    y0 = [0.01, 0.05, 0.15, 0.5, 1.0, 0.1, 0.3]  # Typical fasting/ketosis onset values

    # Model parameters (simplified for simulation)
    # We'll simulate betaHB and acetoacetate with simple kinetics
    # For simplicity, we'll simulate a slow increase in betaHB over time to mimic ketosis onset
    # and acetoacetate in equilibrium with betaHB.
    # We'll also simulate the other gases as baseline with perturbations.

    # Time in hours for slow processes
    t_hours = t_eval / 3600.0

    # Simulate beta-HB: start at 0.15 mM and increase slowly to 0.8 mM over 2 hours
    betaHB_mM = 0.15 + 0.65 * (1 - np.exp(-t_hours / 0.5))  # time constant 0.5 hours
    # Acetoacetate in equilibrium with betaHB (simplified)
    acetoacetate_mM = 0.05 + 0.3 * betaHB_mM  # rough relationship

    # Acetone in blood: assume proportional to acetoacetate (simplified)
    acetone_blood_mM = 0.01 + 0.2 * acetoacetate_mM
    # Breath acetone concentration: assume equilibrium with blood (partition coefficient)
    # Using H_ace from params: dimensionless blood:air partition coeff
    H_ace = 0.001  # from the earlier params
    acetone_ppm = acetone_blood_mM / H_ace  # simplistic: assuming equilibrated alveoli

    # Simulate gut-derived gases: H2, CH4, H2S
    # Baseline levels with some variation and correlation with breathing
    # We'll make them higher during exhalation due to gut perfusion? Actually, gut gases are more constant.
    # We'll just add some random variation.
    base_h2 = 0.5  # ppm
    base_ch4 = 1.0  # ppm
    base_h2s = 0.1  # ppm
    base_nh3 = 0.3  # ppm (in blood, but we'll simulate breath ammonia proportional to blood)

    # Add some random fluctuation (independent of breath)
    h2_ppm = base_h2 + 0.2 * np.random.randn(n_points)
    ch4_ppm = base_ch4 + 0.3 * np.random.randn(n_points)
    h2s_ppm = base_h2s + 0.05 * np.random.randn(n_points)
    nh3_ppm = base_nh3 + 0.1 * np.random.randn(n_points)
    # Ensure non-negative
    h2_ppm = np.maximum(h2_ppm, 0.1)
    ch4_ppm = np.maximum(ch4_ppm, 0.2)
    h2s_ppm = np.maximum(h2s_ppm, 0.01)
    nh3_ppm = np.maximum(nh3_ppm, 0.05)

    # Convert nh3_ppm to ammonia in breath? We'll keep as ppm for consistency.
    # For simplicity, we'll treat nh3_ppm as the breath ammonia concentration.
    ammonia_ppm = nh3_ppm  # we'll keep the name ammonia_ppm for consistency

    # Environmental variables
    # Flow rate: simulate a periodic flow (inspiration negative, expiration positive)
    # FIXED: Correct sign convention - inspiration negative, expiration positive
    breath_cycle = 4.0  # seconds (2 sec inhale, 2 sec exhale)
    breath_phase = (t_eval % breath_cycle) / breath_cycle  # 0 to 1
    # During inhalation (0-0.5): flow negative
    # During exhalation (0.5-1.0): flow positive
    # Use a sinusoidal shape for smoothness
    flow_Lpm = -30 * np.cos(2 * np.pi * breath_phase)  # peak flow 30 L/min, negative during inhale
    # Alternatively: flow_Lpm = 30 * np.sin(2 * np.pi * (breath_phase - 0.25))  # shifted sine
    # Let's use: negative during first half (inhale), positive during second half (exhale)
    flow_Lpm = np.where(breath_phase < 0.5,
                        -60 * (breath_phase - 0.25),  # Negative during inhalation
                        60 * (breath_phase - 0.75))   # Positive during exhalation
    # Ensure reasonable range
    flow_Lpm = np.clip(flow_Lpm, -60, 60)

    # CO2 percentage: inspired ~0%, alveolar ~5%
    # We'll make a smooth transition: low during inspiration, high during expiration
    co2_pct = 0.1 + 4.9 * np.where(breath_phase > 0.5, 1.0, 0.0)  # simple
    # Add some noise
    co2_pct += 0.05 * np.random.randn(n_points)
    co2_pct = np.maximum(co2_pct, 0.0)

    # Humidity and temperature: assume room conditions with small variations
    humidity = 0.5 + 0.1 * np.random.randn(n_points)  # relative fraction
    temperature = 25.0 + 2.0 * np.random.randn(n_points)  # Celsius
    # Clip to reasonable ranges
    humidity = np.clip(humidity, 0.3, 0.8)
    temperature = np.clip(temperature, 20.0, 30.0)

    # Signal stability: random around 1.0
    signal_stability = 1.0 + 0.1 * np.random.randn(n_points)
    signal_stability = np.clip(signal_stability, 0.8, 1.2)

    # Interferents
    # Ethanol: baseline zero, with occasional spikes (simulate drinking)
    ethanol_ppm = np.zeros(n_points)
    # Add a few random spikes
    spike_indices = np.random.choice(n_points, size=5, replace=False)
    ethanol_ppm[spike_indices] = 50.0 + 20 * np.random.randn(len(spike_indices))  # ppm
    ethanol_ppm = np.maximum(ethanol_ppm, 0.0)

    # Breath pH: normally around 7.4, but can vary
    breath_pH = 7.4 + 0.2 * np.random.randn(n_points)
    breath_pH = np.clip(breath_pH, 6.5, 8.0)

    # VOCs (diet-derived volatiles): baseline low, with occasional spikes
    voc_ppm = 0.1 + 0.05 * np.random.randn(n_points)
    voc_ppm = np.maximum(voc_ppm, 0.0)

    # Dead space volume model (physiologically grounded)
    # Anatomical dead space ~2.25 mL/kg, tidal volume ~6-7 mL/kg
    # Using 70kg adult: Vd_anat = 157.5 mL, Vtidal = 420-490 mL
    Vd_anat = 157.5  # mL (anatomical dead space)
    Vtidal = np.mean([420, 490])  # mL (average tidal volume)

    # Dynamic dead space increases with flow rate (based on Bates 2009)
    k_flow = 0.3  # mL/(L/min) flow-dependent dead space coefficient
    # Use absolute value of flow_Lpm because dead space increases with flow magnitude regardless of direction
    Vd_dynamic = k_flow * np.maximum(0, np.abs(flow_Lpm) - 5)  # Additional dead space above 5 L/min
    Vd_total = Vd_anat + Vd_dynamic  # Total dead space

    # Breath capture quality factor: fraction of alveolar air in sample
    # We'll compute a continuous quality factor based on the breath phase and dead space fraction
    # During inhalation: quality low (inspired air mixed with dead space)
    # During exhalation: quality increases as we exhale alveolar air
    # We'll use a simple model: quality = (volume exhaled - dead space) / tidal volume, clipped
    # But we need a continuous signal.

    # Let's compute the exhaled volume relative to the breath cycle.
    # We'll define a variable that represents the fraction of the exhalation cycle that has occurred.
    # During inhalation (0-0.5), we set quality to 0 (or low).
    # During exhalation (0.5-1.0), we compute the fraction of exhalation completed.
    # We'll also account for dead space.

    # We'll create an array for the normalized exhalation progress (0 at start of exhalation, 1 at end)
    exhalation_progress = np.zeros_like(breath_phase)
    mask_exhalation = breath_phase >= 0.5
    exhalation_progress[mask_exhalation] = (breath_phase[mask_exhalation] - 0.5) / 0.5  # 0 to 1

    # The ideal alveolar fraction if no dead space: during inhalation 0, during exhalation exhalation_progress
    # With dead space, the fraction of alveolar air in the mixture is:
    #   V_alveolar / (V_alveolar + Vd) where V_alveolar is the volume of pure alveolar air exhaled so far.
    # We'll approximate: during exhalation, the volume of alveolar air exhaled is proportional to exhalation_progress * Vtidal
    # But we also have dead space mixing.

    # Simple model: quality factor = (exhalation_progress * Vtidal) / (exhalation_progress * Vtidal + Vd_total)
    # This gives 0 at start of exhalation and increases to Vtidal/(Vtidal+Vd_total) at end.
    # We'll also set quality to 0 during inhalation.

    quality_factor = np.zeros_like(t_eval)
    # During inhalation, quality factor remains 0 (inspired air is mostly dead space)
    # During exhalation:
    with np.errstate(divide='ignore', invalid='ignore'):
        quality_factor[mask_exhalation] = (exhalation_progress[mask_exhalation] * Vtidal) / \
                                          (exhalation_progress[mask_exhalation] * Vtidal + Vd_total[mask_exhalation])
    # Clip to [0,1]
    quality_factor = np.clip(quality_factor, 0.0, 1.0)

    # Add breath-to-breath variability in quality (respiratory irregularity)
    quality_noise = 0.05 * np.random.randn(n_points)  # 5% std dev
    quality_factor += quality_noise
    quality_factor = np.clip(quality_factor, 0.0, 1.0)

    # FIXED: Apply quality factor to simulate imperfect breath capture
    # The measured gas concentration is a mixture of alveolar gas and dead space gas
    # We'll assume dead space gas concentration is zero for simplicity (conservative estimate)
    # This means measured concentration = true concentration * quality_factor
    acetone_ppm *= quality_factor
    methane_ppm = ch4_ppm * quality_factor  # FIXED: was missing assignment
    h2_ppm = h2_ppm * quality_factor
    h2s_ppm = h2s_ppm * quality_factor
    # ammonia_ppm already defined from nh3_ppm
    ammonia_ppm = ammonia_ppm * quality_factor

    # Environmental variables remain unaffected by breath capture quality
    # (they are properties of the sampling process/environment)

    # Create output dictionary
    data = {
        'time_seconds': t_eval,
        'time_minutes': t_eval / 60,
        'time_hours': t_eval / 3600,
        'acetone_ppm': acetone_ppm,
        'ammonia_ppm': ammonia_ppm,
        'methane_ppm': methane_ppm,
        'h2_ppm': h2_ppm,
        'h2s_ppm': h2s_ppm,
        'flow_Lpm': flow_Lpm,
        'co2_pct': co2_pct,
        'humidity': humidity,
        'temperature': temperature,
        'signal_stability': signal_stability,
        'ethanol_ppm': ethanol_ppm,  # interferent
        'breath_pH': breath_pH,      # affects partitioning
        'voc_ppm': voc_ppm,          # diet-derived volatiles interferent
        'quality_factor': quality_factor,  # continuous breath-capture quality
        'betaHB_mM': betaHB_mM,      # Ground truth for validation
        'acetoacetate_mM': acetoacetate_mM,
    }

    return data


def save_simulated_data(output_dir='data', duration_hours=2, sample_rate_hz=1):
    """Generate and save the simulated dataset."""
    os.makedirs(output_dir, exist_ok=True)
    data = generate_breath_simulation(duration_hours=duration_hours, sample_rate_hz=sample_rate_hz)

    # Save as compressed numpy file
    output_file_npz = os.path.join(output_dir, 'breath_simulation.npz')
    np.savez_compressed(output_file_npz, **data)

    # Also save as CSV for easy inspection
    import pandas as pd
    df = pd.DataFrame({
        'time_minutes': data['time_minutes'],
        'acetone_ppm': data['acetone_ppm'],
        'ammonia_ppm': data['ammonia_ppm'],
        'methane_ppm': data['methane_ppm'],
        'h2_ppm': data['h2_ppm'],
        'h2s_ppm': data['h2s_ppm'],
        'flow_Lpm': data['flow_Lpm'],
        'co2_pct': data['co2_pct'],
        'humidity': data['humidity'],
        'temperature': data['temperature'],
        'signal_stability': data['signal_stability'],
        'ethanol_ppm': data['ethanol_ppm'],
        'breath_pH': data['breath_pH'],
        'voc_ppm': data['voc_ppm'],
        'quality_factor': data['quality_factor'],
        'betaHB_mM': data['betaHB_mM']
    })
    output_file_csv = os.path.join(output_dir, 'breath_simulation.csv')
    df.to_csv(output_file_csv, index=False)

    print(f"Saved simulated data to:\n  {output_file_npz}\n  {output_file_csv}")
    return output_file_npz


if __name__ == "__main__":
    save_simulated_data()