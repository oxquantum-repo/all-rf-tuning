import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize, curve_fit
from scipy.signal import find_peaks, savgol_filter


def gaussian_pdf(data, mu, sigma):
    P = (1/(np.sqrt(2*np.pi * sigma**2))) * np.exp(-0.5 * ((data - mu) / sigma) ** 2)
    return P
    

def calculate_likelihood(data, mu, sigma):
    likelihood = gaussian_pdf(data, mu, sigma)
    log_likelihood = np.log(likelihood)
    total_log_likelihood = np.sum(log_likelihood)
    return total_log_likelihood


def MLE_single(x, data):
    mu, sigma = x
    likelihood = gaussian_pdf(data, mu, sigma)
    log_likelihood = np.log(likelihood)
    total_log_likelihood = np.sum(log_likelihood)
    return -total_log_likelihood


def MLE_double(x, data, sep_peaks, sep_double):
    w, mu1, mu2, sigma1, sigma2 = x
    likelihood_1 = gaussian_pdf(data, mu1, sigma1)  # blocked
    likelihood_2 = gaussian_pdf(data, mu2, sigma2)  # unblocked
    log_likelihood = np.log(w * likelihood_1 + (1 - w) * likelihood_2)
    total_log_likelihood = np.sum(log_likelihood)

    constraint_lower = np.abs(mu2 - mu1) - ((1 - sep_double) * sep_peaks)
    constraint_upper = np.abs(mu2 - mu1) - ((1 + sep_double) * sep_peaks)
    if constraint_lower < 0:
        total_log_likelihood = total_log_likelihood * (1e3 * (constraint_lower / sep_peaks) ** 2 + 1)  # lorentzian

    elif constraint_upper > 0:
        total_log_likelihood = total_log_likelihood * (1e3 * (constraint_upper / sep_peaks) ** 2 + 1)

    return -total_log_likelihood


def free_GMM(x, data, sep_peaks, sep_double):
    w, mu1, mu2, sigma1, sigma2 = x
    likelihood_1 = gaussian_pdf(data, mu1, sigma1)  # blocked
    likelihood_2 = gaussian_pdf(data, mu2, sigma2)  # unblocked
    log_likelihood = np.log(w * likelihood_1 + (1 - w) * likelihood_2)
    total_log_likelihood = np.sum(log_likelihood)

    constraint_lower = np.abs(mu2 - mu1) - ((1 - sep_double) * sep_peaks)

    if constraint_lower < 0:
        total_log_likelihood = total_log_likelihood * (1e3 * (constraint_lower / sep_peaks) ** 2 + 1)  # lorentzian

    return -total_log_likelihood
    

def mask_outliers(array_of_scores, array_of_valid_measurements): 
    ''' 
        Use Median Absolute Deviation (MAD) instead of t- or z-score, because: 
        1. MAD is totally immune to the sample size
        2. With a breakdown point of 0.5, the median is the location estimator that has the highest breakdown point.
        https://www.sciencedirect.com/science/article/pii/S0022103113000668
        Masking outliers should only beneficial if many samples are used in the routine.
    '''
    valid_scores = array_of_scores[array_of_valid_measurements]
    M = np.median(valid_scores)  # median
    D = valid_scores - M  # deviation
    MAD = 1.4826 * np.median(np.abs(D))
    #lower_bound = M - 2 * MAD  # use 3 for very conservative, 2.5 for moderately conservative, 2 for poorly conservative
    #upper_bound = M + 2 * MAD
    lower_bound = M - 1
    upper_bound = M + 1

    outlier_mask = (array_of_scores >= lower_bound) & (array_of_scores <= upper_bound)  # true entries should be included in score calculation

    return outlier_mask
    

def compute_score(bottom_calibrations, bottom_unblockeds, bottom_pulseds, top_callibrations, top_unblockeds, top_pulseds, abs_detunings, shift_single, sep_double, spread, min_w):
    '''
        shift_single: allowed shift in mean for MLE estimate of single gaussian, as fraction of separation
        sep_double: allowed change in separation of means from references for MLE estimate of double gaussian, as fraction of separation
        spread: allowed change in standard deviation, as percentage
        min_w: minimum weight to be assigned to blocked counts when fitting GMM
    '''
    
    sample_pairs = len(bottom_calibrations)
    BICs = np.empty((sample_pairs, 4))
    outcome_matrix = np.zeros((sample_pairs, 4))
    scores = np.empty((sample_pairs, 1))
    scores_top = np.empty(sample_pairs)
    scores_bot = np.empty(sample_pairs)
    outcome_array = np.empty(sample_pairs)
    means = []
    stds = []

    valid_measurements = np.ones((sample_pairs, 1), dtype=bool)

    # sensor values are converted to mV to make the weights, means and stds a similar order of magnitude.
    for i in range(len(bottom_calibrations)):
        shots = len(top_pulseds[i])

        # Fit of sensor slope caused by detuning sweep
        T_slope, T_c = np.polyfit(-abs_detunings[i][30:70], 1e3*top_callibrations[i][30:70], 1)  # 1e3 to be adjusted based on data scale to keep all log-likelihoods negative
        B_slope, B_c = np.polyfit(abs_detunings[i][30:70], 1e3*bottom_calibrations[i][30:70], 1)

        # Extraction of unblocked reference and extrapolation of blocked references
        T_ref_u, T_std = norm.fit(1e3*top_unblockeds[i]) # Top unblocked reference
        T_ref_b = -B_slope * abs_detunings[i][-1] + B_c  # Top blocked reference
        B_ref_u, B_std = norm.fit(1e3*bottom_unblockeds[i])  # Bottom unblocked reference
        B_ref_b = T_slope * abs_detunings[i][-1] + T_c  # Bottom blocked reference
        means.append([1e-3*B_ref_u, 1e-3*B_ref_b, 1e-3*T_ref_u, 1e-3*T_ref_b])
        stds.append([1e-3*B_std, 1e-3*B_std, 1e-3*T_std, 1e-3*T_std])

        # Set bounds for Maximum Likelihood estimation
        sep_T = np.abs(T_ref_u - T_ref_b)  # separation between blocked and unblocked top
        sep_B = np.abs(B_ref_u - B_ref_b)  # separation between blocked and unblocked bottom
        sep_peaks = (sep_T + sep_B) / 2  # allow them to be the same, which should be true since separation should be identical if sensor is linear
        initial_unblocked_T = [T_ref_u, T_std]
        initial_unblocked_B = [B_ref_u, B_std]
        initial_blocked_T = [0.2, T_ref_b, T_ref_u, T_std, T_std]
        initial_blocked_B = [0.2, B_ref_b, B_ref_u, B_std, B_std]
        bounds_unblocked_T = [(T_ref_u - shift_single * sep_peaks, T_ref_u + shift_single * sep_peaks), ((1 - spread) * T_std, (1 + spread) * T_std)]
        bounds_unblocked_B = [(B_ref_u - shift_single * sep_peaks, B_ref_u + shift_single * sep_peaks), ((1 - spread) * B_std, (1 + spread) * B_std)]
        bounds_blocked_T = [(min_w, 1), 
                            (T_ref_b - shift_single * sep_peaks, T_ref_b + shift_single * sep_peaks), 
                            (T_ref_u - shift_single * sep_peaks, T_ref_u + shift_single * sep_peaks), 
                            ((1 - spread) * T_std, (1 + spread) * T_std), 
                            ((1 - spread) * T_std, (1 + spread) * T_std)]
        bounds_blocked_B = [(min_w, 1), 
                            (B_ref_b - shift_single * sep_peaks, B_ref_b + shift_single * sep_peaks), 
                            (B_ref_u - shift_single * sep_peaks, B_ref_u + shift_single * sep_peaks), 
                            ((1 - spread) * B_std, (1 + spread) * B_std), 
                            ((1 - spread) * B_std, (1 + spread) * B_std)]
       
        opts = {'maxcor': 10,
                'ftol': 2.220446049250313e-09,
                'gtol': 1e-5,
                'eps': 1e-8,
                'maxiter': 15000,
                'maxls': 20,
                'disp': 0}

        # compute maximum likelihood estimates
        result_unblocked_T = minimize(MLE_single, x0=initial_unblocked_T, args=1e3*top_pulseds[i], method='L-BFGS-B', bounds=bounds_unblocked_T, options=opts)
        L_unblocked_T = -result_unblocked_T.fun
    
        result_blocked_T = minimize(MLE_double, x0=initial_blocked_T, args=(1e3*top_pulseds[i], sep_peaks, sep_double), method='L-BFGS-B', bounds=bounds_blocked_T,  options=opts)
        w_T = result_blocked_T.x[0]
        L_blocked_T = -result_blocked_T.fun
    
        result_unblocked_B = minimize(MLE_single, x0=initial_unblocked_B, args=1e3*bottom_pulseds[i], method='L-BFGS-B', bounds=bounds_unblocked_B, options=opts)
        L_unblocked_B = -result_unblocked_B.fun
    
        result_blocked_B = minimize(MLE_double, x0=initial_blocked_B, args=(1e3*bottom_pulseds[i], sep_peaks, sep_double), method='L-BFGS-B', bounds=bounds_blocked_B,  options=opts)
        w_B = result_blocked_B.x[0]
        L_blocked_B = -result_blocked_B.fun
    
        L_nothing = L_unblocked_T + L_unblocked_B
        L_PSB_T = L_blocked_T + L_unblocked_B
        L_PSB_B = L_unblocked_T + L_blocked_B
        
        # re-calculate blocked likelihoods assuming weights in both directions should be equal to get true latching likelihood
        if w_T > w_B:
            initial_blocked_B = [w_T, B_ref_b, B_ref_u, B_std, B_std]
            bounds_blocked_B = [(w_T, w_T), 
                                (B_ref_b - shift_single * sep_peaks, B_ref_b + shift_single * sep_peaks), 
                                (B_ref_u - shift_single * sep_peaks, B_ref_u + shift_single * sep_peaks), 
                                ((1 - spread) * B_std, (1 + spread) * B_std), 
                                ((1 - spread) * B_std, (1 + spread) * B_std)]
            result_blocked_B = minimize(MLE_double, x0=initial_blocked_B, args=(1e3*bottom_pulseds[i], sep_peaks, sep_double), method='L-BFGS-B', bounds=bounds_blocked_B,  options=opts)
            L_blocked_B = -result_blocked_B.fun
    
        elif w_B > w_T:
            initial_blocked_T = [w_B, T_ref_b, T_ref_u, T_std, T_std]
            bounds_blocked_T = [(w_B, w_B), 
                                (T_ref_b - shift_single * sep_peaks, T_ref_b + shift_single * sep_peaks), 
                                (T_ref_u - shift_single * sep_peaks, T_ref_u + shift_single * sep_peaks), 
                                ((1 - spread) * T_std, (1 + spread) * T_std), 
                                ((1 - spread) * T_std, (1 + spread) * T_std)]
            result_blocked_T = minimize(MLE_double, x0=initial_blocked_T, args=(1e3*top_pulseds[i], sep_peaks, sep_double), method='L-BFGS-B', bounds=bounds_blocked_T,  options=opts)
            L_blocked_T = -result_blocked_T.fun
    
        L_latching = L_blocked_T + L_blocked_B
    
        BIC_nothing = np.log(2*shots) * 4 - 2 * L_nothing
        BIC_PSB_T = np.log(2*shots) * 7 - 2 * L_PSB_T
        BIC_PSB_B = np.log(2*shots) * 7 - 2 * L_PSB_B
        BIC_latching = np.log(2*shots) * 9 - 2 * L_latching  # 9 instead of 10 parameters, because weight is shared
        BICs[i] = np.array([BIC_nothing, BIC_PSB_T, BIC_PSB_B, BIC_latching])

        min_idx = np.argmin(BICs[i]) # Most likely model for the i-pair
        outcome_matrix[i, min_idx] = 1
        outcome_array[i] = np.argmax(outcome_matrix[i])

        
        ## add code with min_w = 0 to get scores
        initial_blocked_T = [min_w, T_ref_b, T_ref_u, T_std, T_std]
        initial_blocked_B = [min_w, B_ref_b, B_ref_u, B_std, B_std]
        bounds_blocked_T = [(0, 1), 
                            (T_ref_b - shift_single * sep_peaks, T_ref_b + shift_single * sep_peaks), 
                            (T_ref_u - shift_single * sep_peaks, T_ref_u + shift_single * sep_peaks), 
                            ((1 - spread) * T_std, (1 + spread) * T_std), 
                            ((1 - spread) * T_std, (1 + spread) * T_std)]
        bounds_blocked_B = [(0, 1), 
                            (B_ref_b - shift_single * sep_peaks, B_ref_b + shift_single * sep_peaks), 
                            (B_ref_u - shift_single * sep_peaks, B_ref_u + shift_single * sep_peaks), 
                            ((1 - spread) * B_std, (1 + spread) * B_std), 
                            ((1 - spread) * B_std, (1 + spread) * B_std)]

        result_blocked_T = minimize(MLE_double, x0=initial_blocked_T, args=(1e3*top_pulseds[i], sep_peaks, sep_double), method='L-BFGS-B', bounds=bounds_blocked_T, options=opts)
        w_T = result_blocked_T.x[0]
    
        result_blocked_B = minimize(MLE_double, x0=initial_blocked_B, args=(1e3*bottom_pulseds[i], sep_peaks, sep_double), method='L-BFGS-B', bounds=bounds_blocked_B,  options=opts)
        w_B = result_blocked_B.x[0]

        scores[i] = (w_T - w_B)  # do not take absolute value to distinguish between top and bottom blockade scores
        scores_top[i] = w_T
        scores_bot[i] = w_B
        
        # do checks, and overwrite scores if they are not passed
        smallest_u = min(T_ref_b, T_ref_u, B_ref_b, B_ref_u)
        largest_u = max(T_ref_b, T_ref_u, B_ref_b, B_ref_u)
        smallest_std = min(T_std, B_std)
        largest_std = max(T_std, B_std)
        T_diff_u = np.abs(T_ref_u - T_ref_b)
        B_diff_u = np.abs(B_ref_u - B_ref_b)
        smallest_diff_u = min(T_diff_u, B_diff_u)
        largest_diff_u = max(T_diff_u, B_diff_u)

        # check callibration
        if (largest_diff_u >= (1+shift_single) * smallest_diff_u) or (largest_std >= (1+spread) * smallest_std):
            valid_measurements[i] = False

        else:
            # check pulsing
            free_initial_T = [0.5, T_ref_b, T_ref_u, T_std, T_std]
            free_initial_B = [0.5, B_ref_b, B_ref_u, B_std, B_std]
            free_bounds = [(0, 1), 
                           (smallest_u - largest_diff_u, largest_u + largest_diff_u), 
                           (smallest_u - largest_diff_u, largest_u + largest_diff_u), 
                           (smallest_std * (1-spread), largest_std * (1+spread)), 
                           (smallest_std * (1-spread), largest_std * (1+spread))]
            
            free_fit_T = minimize(free_GMM, x0=free_initial_T, args=(1e3*top_pulseds[i], largest_diff_u, sep_double), method='L-BFGS-B', bounds=free_bounds,  options=opts)
            free_fit_B = minimize(free_GMM, x0=free_initial_B, args=(1e3*bottom_pulseds[i], largest_diff_u, sep_double), method='L-BFGS-B', bounds=free_bounds,  options=opts)
    
            free_T_w, free_T_blocked_u, free_T_unblocked_u, free_T_blocked_std, free_T_unblocked_std = free_fit_T.x[0], free_fit_T.x[1], free_fit_T.x[2], free_fit_T.x[3], free_fit_T.x[4]
            free_B_w, free_B_blocked_u, free_B_unblocked_u, free_B_blocked_std, free_B_unblocked_std = free_fit_B.x[0], free_fit_B.x[1], free_fit_B.x[2], free_fit_B.x[3], free_fit_B.x[4]
            
            T_free_diff_u = np.abs(free_T_blocked_u - free_T_unblocked_u)
            B_free_diff_u = np.abs(free_B_blocked_u - free_B_unblocked_u)

            # not passed if gaussians are fitted in one of the pulsing directions that are separated by more than 50% of the references, and the weight is greater than 0.1
            if np.any((np.array([T_free_diff_u, B_free_diff_u]) >= (1 + shift_single) * largest_diff_u) & (np.array([free_T_w, free_B_w]) >= min_w)):
                valid_measurements[i] = False
                
            else:
                pass

    outlier_mask = mask_outliers(scores, valid_measurements)  # False means they should be rejected
    overall_score = np.abs(np.mean(scores[outlier_mask & valid_measurements])) # overall score handles w_T > w_B or w_B > w_T due to sign

    return BICs, outcome_matrix, scores, outlier_mask, valid_measurements, scores_top, scores_bot, overall_score, means, stds


########################################################################################################################
# Functions for detecting oscillations

def damped_sinusoid(t, A, freq, phi, decay, slope, offset):
    """Model function for a damped sinusoid with a linear trend."""
    return (A * np.sin(2 * np.pi * freq * t + phi)) * np.exp(-t / decay) + offset + slope * t


def damped_double_sinusoid(t, A1, freq1, phi1, A2, freq2, phi2, decay, slope, offset):
    """Model function for a damped double sinusoid with a linear trend."""
    return (A1 * np.sin(2 * np.pi * freq1 * t + phi1) + A2 * np.sin(2 * np.pi * freq2 * t + phi2)) * np.exp(
        -t / decay) + offset + slope * t


def analyze_oscillations(times, signals, threshold=0.5, prominence=0.2, plot=True, filter_data=True, return_fit=False):
    """
    Analyze if there are oscillations in the experimental data within a given frequency range.

    Parameters:
        times (array): Time values (in seconds).
        signals (array): Signal values corresponding to the time values.
        threshold (float): Minimum normalized amplitude in the FFT to consider significant oscillations.
        plot (bool): Whether to plot the results for visualization.
        return_fit (bool): Also return the fit parameters and whether the double sinusoid was used.

    Returns:
        bool: True if oscillations are detected, False otherwise.
        list: Frequencies where oscillations are detected (in MHz).
    """

    # Calculate time step and sampling frequency
    dt = np.mean(np.diff(times))
    fs = 1 / dt

    # Perform FFT
    fft_values = np.fft.fft(signals - np.mean(signals))
    fft_freqs = np.fft.fftfreq(len(signals), d=dt)

    # Take the positive frequencies and normalize the amplitude
    positive_freqs = fft_freqs[fft_freqs > 0]
    positive_amplitudes = np.abs(fft_values[fft_freqs > 0])
    normalized_amplitudes = positive_amplitudes / max(positive_amplitudes)

    # Select frequencies within the desired range
    freq_min, freq_max = 1.5 / np.max(times), 1 / (
                2 * np.diff(times)[0]) / 2.1  # this with the current sampling rate gives 60MHz
    mask = (positive_freqs >= freq_min) & (positive_freqs <= freq_max)
    masked_freqs = positive_freqs[mask]
    masked_amplitudes = normalized_amplitudes[mask]

    # Detect peaks in the FFT spectrum
    peaks, _ = find_peaks(masked_amplitudes, height=threshold, prominence=prominence + np.mean(masked_amplitudes))
    oscillation_freqs = masked_freqs[peaks]

    #########################
    # Filtering
    if filter_data == True:
        signals_filtered = savgol_filter(signals, window_length=6, polyorder=3)

    else:
        signals_filtered = signals

    ##############
    # Fitting depending on frequency components

    if len(oscillation_freqs) == 0 or len(oscillation_freqs) > 2:
        double_component = False  # Flag for plotting latter
        r_squared = 0
        # fit_success = False
        popt = [0, 0, 0, 1, 0, np.mean(signals_filtered)]  # Default parameters for a flat signal

    elif len(oscillation_freqs) == 1:
        double_component = False  # Flag for plotting latter
        freq_guess = oscillation_freqs[0]

        if freq_guess < 2e-3:
            min_bound = 0
        else:
            min_bound = freq_guess - 2e-3

        try:
            initial_guess = [np.ptp(signals_filtered) / 2, freq_guess, 1, 800, 0, np.mean(signals_filtered)]
            bounds = ((-70, min_bound, -np.pi, 100, -1 / 200, np.mean(signals_filtered) * 0.8),
                      (70, freq_guess + 2e-3, np.pi, 8000, 1 / 200, np.mean(signals_filtered) * 1.2))
            popt, pcov = curve_fit(damped_sinusoid, times, signals_filtered, p0=initial_guess, bounds=bounds)

            # Calculate the fitted signal
            fitted_signal = damped_sinusoid(times * 1, *popt)

            # Calculate SS_res and SS_tot
            residuals = signals - fitted_signal
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((signals - np.mean(signals)) ** 2)

            # Calculate R^2
            r_squared = 1 - (ss_res / ss_tot)
            # print(f"R^2 value: {r_squared}")

        except (RuntimeError, IndexError):
            r_squared = 0
            popt = [0, 0, 0, 1, 0, np.mean(signals_filtered)]  # Default parameters for a flat signal

    # Double frequency component fit
    elif len(oscillation_freqs) == 2:
        double_component = True  # Flag for plotting latter

        freq_guess1 = oscillation_freqs[0]
        freq_guess2 = oscillation_freqs[1]

        if freq_guess1 < 2e-3:
            min_bound = 0
        else:
            min_bound = freq_guess1 - 2e-3

        try:
            initial_guess = [np.ptp(signals_filtered) / 2, freq_guess1, 1, np.ptp(signals_filtered) / 2, freq_guess2, 1,
                             800, 0, np.mean(signals_filtered)]
            bounds = (
                (-70, min_bound, -2 * np.pi, -70, min_bound, -np.pi, 100, -1 / 200, np.mean(signals_filtered) * 0.8),
                (70, freq_guess1 + 2e-3, np.pi, 70, freq_guess2 + 2e6, 2 * np.pi, 8000, 1 / 200,
                 np.mean(signals_filtered) * 1.2))
            popt, pcov = curve_fit(damped_double_sinusoid, times, signals_filtered, p0=initial_guess, bounds=bounds)

            # Calculate the fitted signal
            fitted_signal = damped_double_sinusoid(times, *popt)

            # Calculate SS_res and SS_tot
            residuals = signals - fitted_signal
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((signals - np.mean(signals)) ** 2)

            # Calculate R^2
            r_squared = 1 - (ss_res / ss_tot)
            # print(f"R^2 value: {r_squared}")

        except (RuntimeError, IndexError):
            r_squared = 0
            popt = [0, 0, 0, 0, 0, 0, 0, 0, np.mean(signals_filtered)]  # Default parameters for a flat signal

    # Determine if frequency component detected
    has_freqcomponent = (len(oscillation_freqs) == 1 or len(oscillation_freqs) == 2)

    # Plot the results
    if plot and double_component == False:
        plt.figure(figsize=(12, 4))
        plt.subplot(2, 1, 1)
        plt.plot(times, signals * 1e6, label="Original Signal")
        plt.plot(times, signals_filtered * 1e6, label="Filtered Signal")
        plt.plot(times, damped_sinusoid(times, *popt) * 1e6, '--', label="Fitted Damped Sinusoid")
        plt.xlabel("Time (ns)")
        plt.ylabel(r"Amplitude ($\mu V$)")
        plt.title(r"Fitted Damped Sinusoid. R^2 value: {}".format(r_squared))
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(positive_freqs * 1e3, normalized_amplitudes, label="FFT Spectrum")
        plt.scatter(masked_freqs[peaks] * 1e3, masked_amplitudes[peaks], color='red', label="Detected Peaks")
        plt.axvspan(freq_min * 1e3, freq_max * 1e3, color='yellow', alpha=0.15, label="Target Range")
        plt.hlines(np.mean(normalized_amplitudes), np.min(positive_freqs * 1e-6), np.max(positive_freqs * 1e-6),
                   ls='--', color='r')
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Normalized Amplitude")
        plt.title("Frequency-Domain Analysis")
        plt.legend()

        plt.tight_layout()
        plt.show()

    if plot and double_component == True:
        plt.figure(figsize=(12, 4))
        plt.subplot(2, 1, 1)
        plt.plot(times, signals, label="Original Signal")
        plt.plot(times, signals_filtered, label="Filtered Signal")
        plt.plot(times, damped_double_sinusoid(times, *popt), '--')
        plt.xlabel("Time (ns)")
        plt.ylabel("Amplitude")
        plt.title(r"Fitted Damped double Sinusoid. R^2 value: {}".format(r_squared))
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(positive_freqs * 1e3, normalized_amplitudes, label="FFT Spectrum")
        plt.scatter(masked_freqs[peaks] * 1e3, masked_amplitudes[peaks], color='red', label="Detected Peaks")
        plt.axvspan(freq_min * 1e3, freq_max * 1e3, color='yellow', alpha=0.15, label="Target Range")
        plt.hlines(np.mean(normalized_amplitudes), np.min(positive_freqs * 1e-6), np.max(positive_freqs * 1e-6),
                   ls='--', color='r')
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Normalized Amplitude")
        plt.title("Frequency-Domain Analysis")
        plt.legend()

        plt.tight_layout()
        plt.show()

    if return_fit:
        return has_freqcomponent, r_squared, popt, double_component
    return has_freqcomponent, r_squared




