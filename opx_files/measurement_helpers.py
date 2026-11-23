import numpy as np
from opx_files.configuration import *

#### HELPERS ####
def minmax_norm(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def PCA_projection(data1, data2):
    assert data1.shape == data2.shape, 'data input pair have different dimensions!'
    data1, data2 = minmax_norm(data1), minmax_norm(data2)
    Z = np.stack((data1.flatten(), data2.flatten()), axis=1)
    mean = np.mean(Z, axis=0)
    centred_data = Z - mean
    covariance_matrix = np.cov(centred_data.T)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    principal_components = eigenvectors[:, sorted_indices]
    projected_data = centred_data @ principal_components
    reshaped_projected_data = projected_data[:, 0].reshape(data1.shape)
    return reshaped_projected_data


def mV_2_opx_V(desired_mV, gain=5, attenuation=-32):
    opx_volts = (desired_mV * 1e-3) / (gain * 10**(attenuation /20))
    return opx_volts


def opx_V_2_mV(opx_volts, gain=5, attenuation=-32):
    mv = opx_volts * (gain * 10**(attenuation /20)) * 1e3
    return mv


def px_2_mV(idx, scan_range, set_point, resolution):
    step = scan_range / (resolution - 1)
    voltage = (set_point - scan_range / 2) + idx * step
    return voltage


def get_labels(scan_range: list | float | int, set_point: list, resolution: list | float | int):
    if isinstance(resolution, list):
        ticks_x = np.array([0, (resolution[0] -1 )//2, (resolution[0] -1)])
        ticks_y = np.array([0, (resolution[1] -1 )//2, (resolution[1] -1)])
        ticks = [ticks_x, ticks_y]

    elif isinstance(resolution, int | float):
        ticks_x = np.array([0, (resolution -1 )//2, (resolution -1)])
        ticks_y = ticks_x
        ticks = ticks_x

    if isinstance(scan_range, list):
        x_labels = px_2_mV(ticks_x, scan_range[0], set_point[0], resolution)
        y_labels = px_2_mV(ticks_y, scan_range[1], set_point[1], resolution)

    elif isinstance(scan_range, int | float):
        x_labels = px_2_mV(ticks, scan_range, set_point[0], resolution)
        y_labels = px_2_mV(ticks, scan_range, set_point[1], resolution)

    else:
        print("Passed scan range must be a list, float or int")

    return ticks, x_labels, y_labels


def generate_slow_axis_points(voltage_range, n_points):
    """
    Generates an alternating sequence of points around zero, given a number of points and scan size.
    To be used for constructing voltage_offset points for fast scans
    """
    if n_points % 2 == 0:
        # Even number of points, so exclude zero from half_range
        half_range = np.linspace(voltage_range / 2, voltage_range / n_points, n_points // 2)
    else:
        # Odd number of points, include zero
        half_range = np.linspace(voltage_range / 2, 0, n_points // 2 + 1)

    alternating_points = np.empty(n_points)

    # Fill the array alternately with negative and positive values
    alternating_points[0::2] = -half_range[:n_points // 2 + (n_points % 2)]
    alternating_points[1::2] = half_range[:n_points // 2]

    return alternating_points


def unscramble_slow_axis(data):
    """
    Reorders an array whose entries are ordered by the generate_slow_axis_points function
    """
    # Extract elements at even indices
    even_indexed = data[::2]
    # Extract elements at odd indices
    odd_indexed = data[1::2][::-1]

    unscrambled_data = np.concatenate((even_indexed, odd_indexed), axis=0)

    return unscrambled_data


def unscramble_spiral(data):
    """
    Fills a 2d array with 1d data whose entries are generated in a spiral pattern going clockwise, starting from the bottom left corner
    It is assumed the input data was collected along axes of equal size, thereby making the output square
    """
    assert data.ndim == 1, "Only 1d data can be unscrambled from a spiral acquision to make a 2d dataset!"
    n = int(np.sqrt(len(data)))  # Assuming data length is a perfect square
    assert n * n == len(data), "Data length must be a perfect square to form a 2D matrix."
    matrix = np.zeros((n, n))

    # Boundaries for the spiral traversal
    left, right = 0, n - 1
    top, bottom = 0, n - 1
    index = 0

    # Spiral filling
    while left <= right and top <= bottom:
        # Left edge (top to bottom)
        for i in range(bottom, top, -1):
            matrix[i][left] = data[index]
            index += 1

        # Top edge (left to right)
        for i in range(left, right, 1):
            matrix[top][i] = data[index]
            index += 1

        # Right edge (top to bottom)
        for i in range(top, bottom, 1):
            matrix[i][right] = data[index]
            index += 1

        # Bottom edge (right to left)
        for i in range(right, left, -1):
            matrix[bottom][i] = data[index]
            index += 1

        left += 1
        top += 1
        right -= 1
        bottom -= 1

    return matrix


def change_readout_time(time=None):
    if time is None:
        return config['pulses']['reflectometry_readout_pulse']['length']
    else:
        config['pulses']['reflectometry_pulse']['length'] = time  # test element
        config['pulses']['reflectometry_readout_pulse']['length'] = time
        config['integration_weights']['cosine_weights']['cosine'] = [(1.0, time)]
        config['integration_weights']['cosine_weights']['sine'] = [(0.0, time)]
        config['integration_weights']['sine_weights']['cosine'] = [(0.0, time)]
        config['integration_weights']['sine_weights']['sine'] = [(1.0, time)]
        global reflectometry_readout_length
        reflectometry_readout_length = time


def change_readout_amp(amplitude):
    assert amplitude < 0.05, "You planned to set the amplitude above 50 mV, that's too high!"
    if amplitude is None:
        return config['waveforms']['reflect_wf']['sample']
    else:
        config['waveforms']['reflect_wf']['sample'] = amplitude


def update_tof(tof: int):
    global time_of_flight
    time_of_flight = tof
    config['elements']['tank_circuit']['time_of_flight'] = tof


def update_rf_input_offset(offset):
    config['controllers']['con1']['analog_inputs'][2]['offset'] = offset


def change_IF(frequency): # default is 249.5 MHz
    config['elements']['tank_circuit']['intermediate_frequency'] = frequency * u.MHz
    config['elements']['tank_circuit_test']['intermediate_frequency'] = frequency * u.MHz


def change_amp_MW_burst(burst_amp):
    config["waveforms"]["burst_wf"]["sample"] = burst_amp


def remove_sticky_from_config():
    for item in ['initialization', 'idle', 'readout', 'unregistered_value', 'compensation', 'step', 'baked_Op']:
        # For PL_sticky
        keys_to_delete_PL = [key for key in config['elements']['PL_sticky']['operations'] if key.startswith(item)]
        for key in keys_to_delete_PL:
            del config['elements']['PL_sticky']['operations'][key]
            # del config['elements']['PL_sticky_test']['operations'][key]

        # For PM_sticky
        keys_to_delete_PM = [key for key in config['elements']['PM_sticky']['operations'] if key.startswith(item)]
        for key in keys_to_delete_PM:
            del config['elements']['PM_sticky']['operations'][key]
            # del config['elements']['PM_sticky_test']['operations'][key]

        keys_to_delete_PL = [key for key in config['elements']['PL']['operations'] if key.startswith(item)]
        for key in keys_to_delete_PL:
            del config['elements']['PL']['operations'][key]
            # del config['elements']['PL_sticky_test']['operations'][key]

        # For PM_sticky
        keys_to_delete_PM = [key for key in config['elements']['PM']['operations'] if key.startswith(item)]
        for key in keys_to_delete_PM:
            del config['elements']['PM']['operations'][key]
            # del config['elements']['PM_sticky_test']['operations'][key]

    for item in ['PL_sticky_initialization', 'PM_sticky_initialization', 'PL_sticky_idle', 'PM_sticky_idle',
                 'PL_sticky_readout', 'PM_sticky_readout', 'PL_sticky_compensation', 'PM_sticky_compensation',
                 'PL_sticky_unregistered_value', 'PM_sticky_unregistered_value', 'PL_sticky_test', 'PM_sticky_test',
                 'PL_sticky_step', 'PM_sticky_step', 'PL_baked_pulse', 'PM_baked_pulse']:

        # For PL_sticky
        keys_to_delete_PL = [key for key in config['pulses'] if key.startswith(item)]
        for key in keys_to_delete_PL:
            del config['pulses'][key]

        # For PM_sticky
        keys_to_delete_PM = [key for key in config['pulses'] if key.startswith(item)]
        for key in keys_to_delete_PM:
            del config['pulses'][key]

    for item in ['PL_sticky_initialization', 'PM_sticky_initialization', 'PL_sticky_idle', 'PM_sticky_idle',
                 'PL_sticky_readout', 'PM_sticky_readout', 'PL_sticky_compensation', 'PM_sticky_compensation',
                 'PL_sticky_unregistered_value', 'PM_sticky_unregistered_value', 'PL_sticky_test', 'PM_sticky_test',
                 'PL_sticky_step', 'PM_sticky_step', 'PL_baked_wf', 'PM_baked_wf']:

        keys_to_delete = [key for key in config['waveforms'] if key.startswith(item)]
        for key in keys_to_delete:
            del config['waveforms'][key]

    #########################################################################
    for item in ['initialization', 'idle', 'readout', 'unregistered_value', 'compensation', 'step']:
        # For PL_sticky
        keys_to_delete_PL = [key for key in config['elements']['PL_sticky_test']['operations'] if key.startswith(item)]
        for key in keys_to_delete_PL:
            # del config['elements']['PL_sticky']['operations'][key]
            del config['elements']['PL_sticky_test']['operations'][key]

        # For PM_sticky
        keys_to_delete_PM = [key for key in config['elements']['PM_sticky_test']['operations'] if key.startswith(item)]
        for key in keys_to_delete_PM:
            # del config['elements']['PM_sticky']['operations'][key]
            del config['elements']['PM_sticky_test']['operations'][key]


def calculate_distances(samples_top, samples_bottom):
    distances = np.sqrt(np.sum((samples_top - samples_bottom) ** 2, axis=1))
    return distances


## calculate compensation level, given duration, for a 3-point sequence that starts and ends in the same place
def calculate_compensation(timings, levels, compensation_duration):
    channel1_x = [0.0]
    for i, duration in enumerate(timings):
        channel1_x.append(channel1_x[i] + duration)
    channel2_x = channel1_x  # assume plungers are always moved together in time

    channel1_y = [0.0, levels[0], levels[0], levels[2], levels[2], 0.0]
    channel2_y = [0.0, levels[1], levels[1], levels[3], levels[3], 0.0]

    channel1_area = np.trapz(np.array(channel1_y), np.array(channel1_x))
    channel2_area = np.trapz(np.array(channel2_y), np.array(channel2_x))

    compensation_level1 = -channel1_area / compensation_duration
    compensation_level2 = -channel2_area / compensation_duration

    return compensation_level1, compensation_level2
