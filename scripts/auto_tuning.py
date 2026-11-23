import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from scripts.helpers import reconstruction_line, plot_triangles, get_triangle_vertices, get_sampling_points, pulse_endpoints, convert_2_volts
from models.architectures import InterdotCNN, LineCNN, AngleCNN
try:  # needs qualang_tools; only used by get_scan_ranges and get_pulse_lengths
    from opx_files.measurement_helpers import mV_2_opx_V, opx_V_2_mV
except ImportError:
    pass
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist
from scipy.signal import find_peaks
from collections import Counter


def get_quality_score(image):
    """
    Calculate a quality score for an image based on peak detection and thresholding.

    Args:
        image: 2D numpy array representing the image to evaluate

    Returns:
        float: A quality score where higher values indicate better quality
    """
    image = np.array(image).copy()

    row_trace = image[5, :]
    column_trace = image[:, 5]
    peaks_row, _ = find_peaks(row_trace, height=0.12)
    peaks_column, _ = find_peaks(column_trace, height=0.12)

    # must include 1 peak in both row and column
    if (len(peaks_row)==0 or len(peaks_column)==0) or (len(peaks_row) > 1 or len(peaks_column) > 1):
        quality_score = 0

    else:
        lower_bound1 = 0.00474
        upper_bound1 = 0.99916
        lower_bound2 = 0.00122
        upper_bound2 = 0.99916
        minimum_score = 0.085
        image_thresholded1 = image[(image > lower_bound1) & (image < upper_bound1)]
        image_thresholded2 = image[(image > lower_bound2) & (image < upper_bound2)]
        score = np.mean(image_thresholded1) * np.std(image_thresholded2)
        quality_score = score / minimum_score

    return quality_score


def normalise_data(data):
    """
    Normalize data to the range [0, 1].

    Args:
        data: numpy array to be normalized

    Returns:
        numpy array: Normalized data with values between 0 and 1
    """
    data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
    return data_norm


def relabel_interdots(interdot_centers, cluster_labels):
    """
    Relabel interdot centers to ensure consistent ordering, starting from the top-left most cluster.

    Args:
        interdot_centers: numpy array of shape (n, 2) containing the coordinates of interdot centers
        cluster_labels: numpy array of cluster labels

    Returns:
        tuple: (reordered_centers, cluster_labels) - reordered interdot centers and updated cluster labels
    """
    # create dictionary between original labels and new labels
    relabelling_map = {i: None for i in range(len(interdot_centers))}

    interdot_pos = interdot_centers.copy()

    # Identify top left most cluster
    top_left_cluster = np.argmax(interdot_pos[:, 1])

    # assign the key (old cluster label) to the new cluster label (0)
    relabelling_map[top_left_cluster] = 0

    next_interdot = top_left_cluster
    # explore between rays for next interdot to the right
    current_index = 1
    while True:
        try:
            next_interdot = np.where(
                np.all(interdot_centers == nearest_in_angle_ranges(interdot_centers, [[315, 360]], next_interdot),
                       axis=1))[0][0]
            relabelling_map[next_interdot] = current_index
            current_index += 1

        except IndexError:
            # No further coordinate found, select the unremapped coordinate with the highest `y` value
            remaining_indices = [i for i, label in relabelling_map.items() if label is None]
            if not remaining_indices:
                break

            next_interdot = max(remaining_indices, key=lambda i: interdot_centers[i, 1])
            relabelling_map[next_interdot] = current_index
            current_index += 1

    for i in range(len(cluster_labels)):
        cluster_labels[i] = relabelling_map[cluster_labels[i]]

    reordered_centers = [0] * len(interdot_centers)
    for i in range(len(interdot_centers)):
        reordered_centers[relabelling_map[i]] = interdot_centers[i]
    reordered_centers = np.array([list(center) for center in reordered_centers])

    return reordered_centers, cluster_labels


def nearest_in_angle_ranges(coords, angle_ranges, target_idx=None):
    """
    Find the nearest neighbors within specified angle ranges for all points or a single target point.

    Args:
        coords: Array of shape (n, 2), containing the x and y coordinates of all points.
        angle_ranges: List of tuples [(angle_min1, angle_max1), ..., (angle_minN, angle_maxN)],
                      specifying angular ranges in degrees.
        target_idx: Optional, index of the target point to evaluate. If None, evaluates all points.

    Returns:
        If target_idx is None, returns a list of nearest neighbors for all points,
        otherwise returns a list of nearest neighbors for the target point.
    """

    def find_neighbors(point, coords, angle_ranges):
        deltas = coords - point
        mask_self = np.any(deltas != 0, axis=1)  # Exclude the point itself
        angles = np.arctan2(deltas[:, 1], deltas[:, 0])
        angles = np.mod(angles, 2 * np.pi)  # Normalize to [0, 2π]
        neighbors = []

        for angle_min, angle_max in np.radians(angle_ranges):  # Convert ranges to radians
            mask_angle = (angles >= angle_min) & (angles <= angle_max)
            valid_points = coords[mask_self & mask_angle]
            valid_deltas = deltas[mask_self & mask_angle]
            distances = np.linalg.norm(valid_deltas, axis=1)

            if len(distances) > 0:
                nearest_idx = np.argmin(distances)
                neighbors.append(valid_points[nearest_idx])
            else:
                neighbors.append(None)

        return neighbors

    if target_idx is not None:
        # Compute for a single point
        return find_neighbors(coords[target_idx], coords, angle_ranges)[0]
    else:
        # Compute for all points
        return [find_neighbors(point, coords, angle_ranges) for point in coords]


def find_midpoints_of_neighbors(coords, nearest_neighbors):
    """
    Calculate midpoints between pairs of nearest neighbors for each coordinate.

    Args:
        coords: numpy array of shape (n, 2) containing coordinates
        nearest_neighbors: list of nearest neighbors for each coordinate

    Returns:
        list: A list of midpoints for each coordinate, where each entry contains up to two midpoints
    """
    midpoints = []

    for i, point in enumerate(coords):
        # Initialize the list of midpoints for this point
        point_midpoints = []

        # Get the first two neighbors (indices 0 and 1)
        if len(nearest_neighbors[i]) >= 2 and nearest_neighbors[i][0] is not None and nearest_neighbors[i][
            1] is not None:
            neighbor1 = nearest_neighbors[i][0]
            neighbor2 = nearest_neighbors[i][1]
            midpoint1 = np.array([(neighbor1[0] + neighbor2[0]) / 2, (neighbor1[1] + neighbor2[1]) / 2])
            point_midpoints.append(midpoint1)
        else:
            point_midpoints.append(np.array([]))  # Empty array if no valid pair

        # Get the second pair of neighbors (indices 2 and 3)
        if len(nearest_neighbors[i]) >= 4 and nearest_neighbors[i][2] is not None and nearest_neighbors[i][
            3] is not None:
            neighbor3 = nearest_neighbors[i][2]
            neighbor4 = nearest_neighbors[i][3]
            midpoint2 = np.array([(neighbor3[0] + neighbor4[0]) / 2, (neighbor3[1] + neighbor4[1]) / 2])
            point_midpoints.append(midpoint2)
        else:
            point_midpoints.append(np.array([]))  # Empty array if no valid pair

        # Append the midpoints for this point
        midpoints.append(point_midpoints)

    return midpoints


def improve_midpoints(midpoints, coords):
    """
    Improve midpoints by mirroring existing midpoints when one is missing.

    Args:
        midpoints: list of midpoints for each coordinate
        coords: numpy array of shape (n, 2) containing coordinates

    Returns:
        list: Updated midpoints with missing entries filled in by mirroring
    """
    # Iterate over each coordinate and its corresponding midpoints
    for i, point in enumerate(coords):
        # Check if both midpoint entries are empty
        if midpoints[i][0].size == 0 and midpoints[i][1].size == 0:
            # Leave the midpoints empty if both are empty
            continue

        # Check if the first midpoint is empty and the second is not
        if midpoints[i][0].size == 0 and midpoints[i][1].size > 0:
            # Mirror the second midpoint relative to the current point
            mirrored_midpoint = 2 * point - midpoints[i][1]
            midpoints[i][0] = mirrored_midpoint  # Replace the first midpoint with the mirrored one

        # Check if the second midpoint is empty and the first is not
        if midpoints[i][0].size > 0 and midpoints[i][1].size == 0:
            # Mirror the first midpoint relative to the current point
            mirrored_midpoint = 2 * point - midpoints[i][0]
            midpoints[i][1] = mirrored_midpoint  # Replace the second midpoint with the mirrored one

    return midpoints


def find_nearest_valid_neighbor(current_idx, neighbors, midpoints, coords):
    """
    Find the nearest neighbor that has both valid midpoints.
    """
    valid_neighbors = []

    # Iterate over the neighbors of the current coordinate
    for neighbor_coords in neighbors[current_idx]:
        # Find the index of the neighbor in the coords array
        if neighbor_coords is not None:
            # Get the index of the neighbor in the coords array
            neighbor_idx = np.where(np.all(coords == neighbor_coords, axis=1))[0]

            # If the neighbor exists in coords, proceed
            if len(neighbor_idx) > 0:
                neighbor_idx = neighbor_idx[0]

                # Check if both midpoints are valid
                if midpoints[neighbor_idx][0].size > 0 and midpoints[neighbor_idx][1].size > 0:
                    valid_neighbors.append(neighbor_idx)

    if len(valid_neighbors) == 0:
        return None

    # Calculate distances to valid neighbors
    current_point = coords[current_idx]
    distances = []
    for neighbor_idx in valid_neighbors:
        distance = np.linalg.norm(current_point - coords[neighbor_idx])
        distances.append(distance)

    # Return the index of the nearest valid neighbor
    nearest_neighbor_idx = valid_neighbors[np.argmin(distances)]
    return nearest_neighbor_idx


def translate_midpoints_to_current_coordinate(current_idx, nearest_neighbor_idx, coords, midpoints):
    """
    Translate the midpoints from the nearest neighbor to be centered around the current coordinate.
    """
    current_point = coords[current_idx]
    neighbor_point = coords[nearest_neighbor_idx]

    # Calculate offsets
    offset_1 = midpoints[nearest_neighbor_idx][0] - neighbor_point
    offset_2 = midpoints[nearest_neighbor_idx][1] - neighbor_point

    # Translate midpoints
    midpoints[current_idx][0] = current_point + offset_1
    midpoints[current_idx][1] = current_point + offset_2


def handle_empty_midpoints(coords, midpoints, neighbors):
    """
    Iterate over the midpoints list, and if both entries are empty, find the nearest neighbor
    with both valid midpoints and translate the midpoints to be centered around the current coordinate.
    """
    for i, point in enumerate(coords):
        # If both midpoints are empty, process
        if midpoints[i][0].size == 0 and midpoints[i][1].size == 0:
            # Find the nearest neighbor with valid midpoints
            nearest_neighbor_idx = find_nearest_valid_neighbor(i, neighbors, midpoints, coords)
            if nearest_neighbor_idx is not None:
                # Translate midpoints to be centered around the current coordinate
                translate_midpoints_to_current_coordinate(i, nearest_neighbor_idx, coords, midpoints)

    return midpoints


def get_scan_ranges(coordinates, extent):
    """
    Calculate scan ranges for each coordinate based on the distance to the nearest neighbor.

    Args:
        coordinates: numpy array of shape (n, 2) containing coordinates
        extent: float, multiplier to determine the scan range

    Returns:
        list: Scan ranges for each coordinate, limited to the OPX voltage range
    """
    scan_ranges = []

    for i in range(len(coordinates)):
        distances = np.linalg.norm(coordinates - coordinates[i], axis=1)  # Compute distances from point i
        distances[i] = np.inf  # Ignore distance to itself
        scan_range = 2 * np.min(distances) * extent

        if mV_2_opx_V(scan_range) > 0.5:  # range is +-0.5
            scan_ranges.append(opx_V_2_mV(0.49))
        else:
            scan_ranges.append(scan_range)

    return scan_ranges


def get_pulse_lengths(coords, midpoints, extent):
    """
    Calculate pulse lengths and points for each coordinate based on midpoints.

    Args:
        coords: numpy array of shape (n, 2) containing coordinates
        midpoints: list of midpoints for each coordinate
        extent: float, multiplier to determine the pulse length

    Returns:
        tuple: (pulse_lengths, pulse_points) - lengths and endpoints for pulses
    """
    pulse_lengths = []

    for center, midpoint_pair in zip(coords, midpoints):
        distance = (np.linalg.norm(center - midpoint_pair[0]) + np.linalg.norm(center - midpoint_pair[1])) / 2
        length = distance * extent
        if mV_2_opx_V(
                length / 1.2) > 0.5:  # single jump cannot be bigger than +-0.5. Components of diagonal are bigger than length (hence divide by some safety factor)
            pulse_lengths.append(opx_V_2_mV(0.5) * 1.2)
        else:
            pulse_lengths.append(length)

    pulse_points = []

    for coord, midpoints_pair, length in zip(coords, midpoints, pulse_lengths):
        unit_vector = (midpoints_pair[0] - midpoints_pair[1]) / np.linalg.norm((midpoints_pair[0] - midpoints_pair[1]))

        new_midpoints_pair = []
        for midpoint in midpoints_pair:
            if midpoint[1] > coord[1]:
                adjusted_point = coord + unit_vector * length
            else:
                adjusted_point = coord - unit_vector * length
            new_midpoints_pair.append(adjusted_point)

        pulse_points.append(new_midpoints_pair)

    return pulse_lengths, pulse_points


def distance_matrix(data):
    """
    Calculate the pairwise Euclidean distance matrix for a set of points.

    Args:
        data: numpy array of shape (n, d) containing n points in d dimensions

    Returns:
        numpy array: A square matrix of shape (n, n) containing pairwise distances
    """
    num_points = data.shape[0]
    distances = np.zeros((num_points, num_points))
    for i in range(num_points):
        distances[i] = np.sqrt(np.sum((data - data[i])**2, axis=1))
    return distances


def filter_by_neighbours(data, neighborhood_radius, min_neighbors):
    """
    Filter data points by removing those with fewer than min_neighbors within neighborhood_radius.

    Args:
        data: numpy array of shape (n, d) containing n points in d dimensions
        neighborhood_radius: float, radius within which to count neighbors
        min_neighbors: int, minimum number of neighbors required to keep a point

    Returns:
        tuple: (filtered_data, discarded_data, single_cluster) - filtered points, discarded points, 
               and a boolean indicating if the remaining points form a single cluster
    """
    n_removed_points = 1
    filtered_data = data
    while n_removed_points != 0:  # iteratively remove points until all remaining remaining clusters have minimum nearest neighbours within radius
        starting_n = len(filtered_data)
        distances = distance_matrix(filtered_data)
        num_neighbors = np.sum(distances <= neighborhood_radius, axis=1) - 1  # Exclude self-distance
        filtered_data_indices = np.where(num_neighbors >= min_neighbors)[0]
        filtered_data = filtered_data[filtered_data_indices]
        ending_n = len(filtered_data)
        n_removed_points = starting_n - ending_n

    single_cluster = False
    if np.mean(distance_matrix(filtered_data)) < 8.8:  # value for two clusters, with 16 and 3 points respectively, separated by 4 pixels
        single_cluster = True

    set1 = set(map(tuple, data))
    set2 = set(map(tuple, filtered_data))
    discarded_data = np.array(list(set1 - set2))

    return filtered_data, discarded_data, single_cluster


def silhoutte_scores(data, min_clusters, max_clusters):
    """
    Calculate silhouette scores for different numbers of clusters.

    Args:
        data: numpy array of shape (n, d) containing n points in d dimensions
        min_clusters: int, minimum number of clusters to try
        max_clusters: int, maximum number of clusters to try

    Returns:
        list: Silhouette scores for each number of clusters from min_clusters to max_clusters
    """
    silhouette_scores = []
    num_clusters = []
    for k in range(min_clusters, max_clusters + 1):  # Trying different numbers of clusters from 2 to 10
        kmeans = KMeans(init='k-means++', n_clusters=k, n_init=10, max_iter=50)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        num_clusters.append(k)

    return silhouette_scores


def dbi_indices(data, min_clusters, max_clusters):
    """
    Calculate Davies-Bouldin indices for different numbers of clusters.

    Args:
        data: numpy array of shape (n, d) containing n points in d dimensions
        min_clusters: int, minimum number of clusters to try
        max_clusters: int, maximum number of clusters to try

    Returns:
        list: Davies-Bouldin indices for each number of clusters from min_clusters to max_clusters
    """
    dbi_indices = []
    num_clusters = []
    for k in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(init='k-means++', n_clusters=k, n_init=10, max_iter=50)
        labels = kmeans.fit_predict(data)
        centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])

        # Calculate intra-cluster distances
        cluster_scatters = np.zeros(k)
        for i in range(k):
            cluster_points = data[labels == i]
            cluster_scatters[i] = np.mean(np.linalg.norm(cluster_points - centroids[i], axis=1))

        # Calculate inter-cluster distances
        distances = euclidean_distances(centroids)
        np.fill_diagonal(distances, np.inf)  # Avoid division by zero for the same cluster
        R = np.zeros((k, k))

        for i in range(k):
            for j in range(k):
                if i != j:
                    R[i, j] = (cluster_scatters[i] + cluster_scatters[j]) / distances[i, j]

        # Compute the Davies-Bouldin Index
        dbi = np.mean(np.max(R, axis=1))

        dbi_indices.append(dbi)
        num_clusters.append(k)

    return dbi_indices


def dunn_indices(data, min_clusters, max_clusters, min_size, max_size):
    """
    Calculate Dunn indices for different numbers of clusters with population constraints.

    Args:
        data: numpy array of shape (n, d) containing n points in d dimensions
        min_clusters: int, minimum number of clusters to try
        max_clusters: int, maximum number of clusters to try
        min_size: int, minimum allowed cluster size
        max_size: int, maximum allowed cluster size

    Returns:
        list: Dunn indices for each number of clusters from min_clusters to max_clusters
    """
    dunn_indices = []
    if len(data) < max_clusters:
        max_clusters = len(data)
    for k in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(init='k-means++', n_clusters=k, n_init=20, max_iter=50)
        labels = kmeans.fit_predict(data)
        cluster_centers = kmeans.cluster_centers_

        ## re-assign points to nearby clusters if populations are unreasonable
        if min_size <= len(data)/k <= max_size:  # only reassign if number of clusters compared to data size allows
            while True:
                cluster_counts = dict(Counter(labels))
                small_clusters = {key for key, val in cluster_counts.items() if val < min_size}
                large_clusters = {key for key, val in cluster_counts.items() if val > max_size}

                if large_clusters:
                    points_to_reassign = []
                    # Collect excess points from large clusters, prioritizing ones further from their centroid
                    for cluster in large_clusters:
                        excess = cluster_counts[cluster] - max_size
                        indices = np.where(labels == cluster)[0]

                        distances = np.linalg.norm(data[indices] - cluster_centers[cluster], axis=1)
                        farthest_indices = indices[np.argsort(-distances)[:int(excess)]]
                        points_to_reassign.extend(farthest_indices)

                    for idx in points_to_reassign:
                        distances = cdist([data[idx]], cluster_centers).flatten()
                        sorted_clusters = np.argsort(distances)  # Order clusters by distance

                        # Donate excess points to nearest cluster that satisfies population requirements
                        for new_label in sorted_clusters:  # iterates over clusters in increasing distance
                            if cluster_counts[new_label] < max_size:
                                # Update cluster size and reassign
                                cluster_counts[labels[idx]] -= 1
                                cluster_counts[new_label] += 1
                                labels[idx] = new_label
                                break

                elif small_clusters:
                    for cluster in small_clusters:
                        recess = min_size - cluster_counts[cluster]  # how many points need to be accepted
                        distances = np.linalg.norm(data - cluster_centers[cluster], axis=1)
                        nearest_indices = np.argsort(distances)

                        # Accept points from nearest cluster that satisfies population requirements
                        reassigned = 0
                        for idx in nearest_indices:
                            if reassigned >= recess:
                                break

                            original_cluster = labels[idx]
                            if cluster_counts[original_cluster] > min_size:
                                # Update cluster size and reassign
                                cluster_counts[original_cluster] -= 1
                                cluster_counts[cluster] += 1
                                labels[idx] = cluster
                                reassigned += 1

                else:  # no clusters that are too big or too small
                    break

        # Compute intra-cluster diameters
        intra_cluster_diameters = []
        for label in range(k):
            cluster_points = data[labels == label]
            distances = cdist(cluster_points, cluster_points)
            intra_cluster_diameters.append(np.max(distances))
        max_intra_diameter = max(intra_cluster_diameters)

        # Compute inter-cluster distances
        inter_cluster_distances = []
        for i, label1 in enumerate(range(k)):
            for j, label2 in enumerate(range(k)):
                if i < j:  # Only consider pairs once
                    cluster1_points = data[labels == label1]
                    cluster2_points = data[labels == label2]
                    # Compute pairwise distances between two clusters
                    distances = cdist(cluster1_points, cluster2_points)
                    inter_cluster_distances.append(np.min(distances))
        min_inter_distance = min(inter_cluster_distances)

        # Compute Dunn Index
        dunn = min_inter_distance / max_intra_diameter

        dunn_indices.append(dunn)

    return dunn_indices


def adjust_centers(k, centroids, stride, x_res, y_res):
    """
    Adjust cluster centers to avoid image edges.

    Args:
        k: int, number of clusters
        centroids: numpy array of shape (k, 2) containing cluster centers
        stride: int, stride used in the scanning process
        x_res: int, x-resolution of the image
        y_res: int, y-resolution of the image

    Returns:
        list: Adjusted box centers for each cluster
    """
    box_centers = []

    for cluster in range(k):
        center_x, center_y = centroids[cluster]
        print('initial:', center_x, center_y)

        # adjust bounding box position if interdot is near image edge
        if center_y > (y_res - 10 - 1.5*stride):  # 10 is half of patch size
            center_y = center_y + 2*stride

        if center_y < (10 + 1.5*stride):
            center_y = center_y - 2*stride

        if center_x > (x_res - 10 - 1.5*stride):
            center_x = center_x + 2*stride

        if center_x < (10 + 1.5*stride):
            center_x = center_x - 2*stride

        print('final:', center_x, center_y)

        box_centers.append([center_x, center_y])

    return box_centers


def run_interdot_finder(image, scan_params, stride=4, threshold=0.5, neighborhood_radius=9, min_neighbors=3, min_clusters=2, max_clusters=10,  min_population=4, max_population=23, model='classification_model_0_31873.pth'):
    """
    Identify interdot regions in an image using a CNN classifier and clustering.

    Args:
        image: 2D numpy array representing the image to analyze
        scan_params: list, parameters for the scan
        stride: int, pixel step size between patches passed for evaluation by the binary interdot classifier
        threshold: float, threshold for positive classification
        neighborhood_radius: float, pixel radius about each positive classification for neighbor evaluation
        min_neighbors: int, minimum number of neighbors required for a positive classification
        min_clusters: int, minimum number of clusters to try
        max_clusters: int, maximum number of clusters to try
        min_population: int, minimum allowed cluster size
        max_population: int, maximum allowed cluster size
        model: str, path to the CNN model file

    Returns:
        tuple: (filtered_data, cluster_centers, cluster_labels, interdot_centers, discarded_data) - 
               filtered points, cluster centers, cluster labels, adjusted interdot centers, and discarded points
    """

    def classify_patch(image):
        input_image = np.expand_dims(image, axis=0)  # Add color channel dimension at index 0
        input_image_copy = input_image.copy()  # removes negative strides after slicing
        X = torch.Tensor(input_image_copy).unsqueeze(0)  # Add batch dimension at index 0
        X = X.to(device)
        with torch.no_grad():
            output = model(X)[0][0].item()
            y = 1/(1+np.exp(-output))  # apply sigmoid activation
        return y

    script_dir = os.path.dirname(os.path.abspath(__file__))  # __file__ points to this file
    model_path = os.path.join(script_dir, model)

    device = torch.device("cpu")
    model = InterdotCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()

    image = np.flipud(image)

    x_true = []
    y_true = []
    x_res, y_res = image.shape
    for row_start in range(0, y_res - 20 + 1, stride):  # patch resolution fixed at 20 due to network input dimension
        for col_start in range(0, x_res - 20 + 1, stride):
            x_coord = col_start + 10
            y_coord = row_start + 10

            sub_image = image[row_start:row_start + 20, col_start:col_start + 20]
            sub_image = normalise_data(sub_image)
            interdot_present = classify_patch(sub_image)

            if interdot_present > threshold:  # use 0.5 as a threshold
                x_true.append(x_coord)
                y_true.append(y_coord)

    # Filter data points based on neighbours criterion
    data = np.column_stack((x_true, y_true))
    filtered_data, discarded_data, single_cluster = filter_by_neighbours(data, neighborhood_radius, min_neighbors)

    # Create clusters
    if len(filtered_data) > 1:  # at least 2 points are needed to make 2 clusters
        if single_cluster:
            optimal_k = 1
            cluster_centers = np.array([np.mean(filtered_data, axis=0)])
            cluster_labels = np.zeros(len(filtered_data))
            interdot_centers = adjust_centers(optimal_k, cluster_centers, stride, x_res, y_res)
        else:
            scores_sil = silhoutte_scores(filtered_data, min_clusters, max_clusters)
            scores_dbi = dbi_indices(filtered_data, min_clusters, max_clusters)
            score_dunn = dunn_indices(filtered_data, min_clusters, max_clusters, min_population, max_population)
            optimal_k = np.argmax(score_dunn) + min_clusters  # determine optimal number of clusters from maximum of Dunn score

            plt.figure(figsize=(6, 3.5))
            plt.plot(np.arange(min_clusters, max_clusters + 1), scores_sil, label='Silhouette Score')
            plt.plot(np.arange(min_clusters, max_clusters + 1), scores_dbi, label='DBI Index')
            plt.plot(np.arange(min_clusters, max_clusters + 1), score_dunn, label='Dunn Index')
            plt.scatter(optimal_k, score_dunn[optimal_k - min_clusters], color='r')
            plt.title('Clustering Metrics')
            plt.xlabel('Clusters')
            plt.ylabel('Score')
            plt.legend(frameon=False)
            plt.show()

            kmeans = KMeans(init='k-means++', n_clusters=optimal_k, n_init=10, max_iter=50)  # Perform clustering
            cluster_labels = kmeans.fit_predict(filtered_data)
            cluster_centers = kmeans.cluster_centers_  # determine cluster centroids (interdot location)
            interdot_centers = adjust_centers(optimal_k, cluster_centers, stride, x_res, y_res)  # get bounding boxes (scan window)
    else:
        print('no interdots could be found')
        return filtered_data, None, None, None, discarded_data

    filtered_data = np.array(convert_2_volts(filtered_data, scan_params))
    cluster_centers = np.array(convert_2_volts(cluster_centers, scan_params))
    discarded_data = np.array([convert_2_volts(discarded_data, scan_params) if len(discarded_data) > 0 else discarded_data][0])
    interdot_centers = np.array(convert_2_volts(interdot_centers, scan_params))

    return filtered_data, cluster_centers, cluster_labels, interdot_centers, discarded_data


def mask_metastable_regions(image, scan_params, n=6, contract=True, factor=0.35, pulse_length='auto'):
    """
    Identify metastable regions in an image using CNN models for line detection and angle prediction.

    Args:
        image: 2D numpy array representing the image to analyze
        scan_params: list, parameters for the scan
        n: int, number of sampling points to generate
        contract: bool, whether to contract the sampling points
        factor: float, contraction factor
        pulse_length: str or float, pulse length to use ('auto' for automatic determination)

    Returns:
        tuple: (line_image, quality_score, vertex_coords, triangle_coords, pulse_offsets, bottom_samples, top_samples) -
               processed line image, quality score, triangle vertices, triangle coordinates, pulse endpoints,
               and sampling points
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))  # __file__ points to this file
    model_path1 = os.path.join(script_dir, '../models/Line_CNN_model.pth')
    model_path2 = os.path.join(script_dir, '../models/Angle_CNN_model.pth')

    device = torch.device("cpu")

    model1 = LineCNN().to(device)
    model1.load_state_dict(torch.load(model_path1, map_location=device, weights_only=False))
    model1.eval()

    model2 = AngleCNN().to(device)
    model2.load_state_dict(torch.load(model_path2, map_location=device, weights_only=False))
    model2.eval()

    image = image[::-1]
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    input_image = image.reshape(1, image.shape[0], image.shape[1])
    X1 = torch.Tensor(input_image).unsqueeze(0)  # Add batch dimension at index 0
    X1 = X1.to(device)
    with torch.no_grad():
        line_image = model1(X1)[0][0]
    quality_score = get_quality_score(line_image)  # above 1 passes threshold

    X2 = line_image.reshape(1, line_image.shape[0], line_image.shape[1])  # adds batch dimension,
    with torch.no_grad():
        angles = model2(X2.unsqueeze(0))[0]  # no colour dimension to remove, only batch

    line_coeffs, _, _ = reconstruction_line(angles.cpu(), image.shape[0])

    vertex_coords = convert_2_volts(get_triangle_vertices(line_coeffs), scan_params)
    triangle_coords = np.array(convert_2_volts(np.array(plot_triangles(line_coeffs)).T, scan_params))  # used only for plotting

    vertex_coords = [[x[0] + scan_params[2]/scan_params[3], x[1] - scan_params[2]/scan_params[3]] for x in vertex_coords]  # fudge factor
    triangle_coords = np.array([[x[0] + scan_params[2]/scan_params[3], x[1] - scan_params[2]/scan_params[3]] for x in triangle_coords])  # fudge factor

    pulse_offsets = pulse_endpoints(vertex_coords, pulse_length)

    top_samples, bottom_samples = get_sampling_points(vertex_coords, n, contract, factor)

    return line_image, quality_score, vertex_coords, triangle_coords, pulse_offsets, bottom_samples, top_samples


def get_metastable_vertices(image, scan_params):
    """
    Extract vertices of metastable regions in an image using CNN models.

    Args:
        image: 2D numpy array representing the image to analyze
        scan_params: list, parameters for the scan

    Returns:
        tuple: (line_image, quality_score, vertex_coords, triangle_coords, interdot_midpoint) -
               processed line image, quality score, triangle vertices, triangle coordinates,
               and the midpoint between vertices
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))  # __file__ points to this file
    model_path1 = os.path.join(script_dir, '../models/Line_CNN_model.pth')
    model_path2 = os.path.join(script_dir, '../models/Angle_CNN_model.pth')

    device = torch.device("cpu")

    model1 = LineCNN().to(device)
    model1.load_state_dict(torch.load(model_path1, map_location=device, weights_only=False))
    model1.eval()

    model2 = AngleCNN().to(device)
    model2.load_state_dict(torch.load(model_path2, map_location=device, weights_only=False))
    model2.eval()

    image = image[::-1]
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    input_image = image.reshape(1, image.shape[0], image.shape[1])
    X1 = torch.Tensor(input_image).unsqueeze(0)  # Add batch dimension at index 0
    X1 = X1.to(device)
    with torch.no_grad():
        line_image = model1(X1)[0][0]
    quality_score = get_quality_score(line_image)  # 0 or 1 depending on goodness score of the line extraction based on mean absolute deviation from 1

    X2 = line_image.reshape(1, line_image.shape[0], line_image.shape[1])  # adds batch dimension,
    with torch.no_grad():
        angles = model2(X2.unsqueeze(0))[0]  # no colour dimension to remove, only batch

    line_coeffs, _, _ = reconstruction_line(angles.cpu(), image.shape[0])

    vertex_coords = convert_2_volts(get_triangle_vertices(line_coeffs), scan_params)
    triangle_coords = np.array(convert_2_volts(np.array(plot_triangles(line_coeffs)).T, scan_params))  # used only for plotting

    vertex_coords = [[x[0] + scan_params[2]/scan_params[3], x[1] - scan_params[2]/scan_params[3]] for x in vertex_coords]  # fudge factor
    triangle_coords = np.array([[x[0] + scan_params[2]/scan_params[3], x[1] - scan_params[2]/scan_params[3]] for x in triangle_coords])  # fudge factor

    interdot_midpoint = [(vertex_coords[0][0] + vertex_coords[2][0])/2, (vertex_coords[0][1] + vertex_coords[2][1])/2] 

    return line_image, quality_score, vertex_coords, triangle_coords, interdot_midpoint
