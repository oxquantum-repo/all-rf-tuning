import numpy as np
import cv2
from skimage.morphology import skeletonize
from plantcv import plantcv as pcv


'''
This script includes helper functions for the:
    - angle parametrisation of transition lines (used to make labels for the Angle CNN)
    - make meta-stable masks (used in the evaluation of the score function for PSB)
    - overlay meta-stable triangles (to visualize the accuracy of the two-stage CNNs)
'''

# Equation of a straight line
def straight_line(x, m, c):
    return m*x + c


def two_line_intersection(m1, m2, c1, c2):
    x0 = (c2 - c1) / (m1 - m2)
    y0 = straight_line(x0, m1, c1)
    return x0, y0


# Perpendicular distance between a point and a line
def point_line_distance(x0, y0, m, c):
    a = -m
    b = 1
    c = -c
    return abs(a*x0 + b*y0 + c) / np.sqrt(a**2 + b **2)


# Acute angle made by two intersecting lines
def line_intersect_angle(m1, m2):
    return np.arctan(np.abs((m1 - m2) / (1 + m1*m2)))


# Slope of line connecting two points
def two_point_slope(x1, y1, x2, y2):
    return (y2-y1)/(x2-x1)


# intersection of a line, defined by two points, with the x-axis
def two_point_x_intercept(x1, y1, x2, y2):
    m = two_point_slope(x1, y1, x2, y2)
    c = get_c(x1, y1, m)
    return -c/m


# Length of a contour between two points
def two_point_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


# Mid-point coordinate between two points
def midpoint(x1, y1, x2, y2):
    return (x1+x2)/2, (y1+y2)/2


# y-intercept of a ray from some point with known slope
def get_c(x0, y0, m):
    return y0 - m*x0


# unit vector for a ray along a given direction (slope) 'm'
def unit_vector(m):
    if m == np.inf:
        return np.array([0, 1])
    else:
        vec = np.array([1, m])
        unit_vec = vec / np.sqrt(1 + m**2)
    return unit_vec


# End point of a ray projected along a slope 'm' with length 'd'
def ray_along_direction(p0, m, d):
    unit_vec = unit_vector(m)
    return p0 + d * unit_vec


# Get the pixel indices that fall along a straight line
def bresenham_line(start, end):
    x1, y1 = start
    x2, y2 = end
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    line = []
    while True:
        line.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    return line


def make_bounding_box(midpoint, box_size, image_size):
    center_row = midpoint[0]
    center_col = midpoint[1]
    half_size = int(box_size // 2)
    start_row = max(center_row - half_size, 0)
    end_row = min(center_row + half_size + 1, image_size)
    start_col = max(center_col - half_size, 0)
    end_col = min(center_col + half_size + 1, image_size)
    row_indices, col_indices = np.meshgrid(np.arange(start_row, end_row), np.arange(start_col, end_col), indexing='ij')
    return row_indices, col_indices


# Converts pixel indices to voltage values
def convert_2_volts(coords, scan_params):
    x_center, y_center, scan_range, steps = scan_params
    x_start = x_center - scan_range / 2
    y_start = y_center - scan_range / 2
    increment = scan_range / steps

    def convert_single(coord):
        x, y = coord
        vx = x_start + (increment * x)
        vy = (y_start + scan_range) - (increment * y)
        return [vx, vy]

    if isinstance(coords, list):
        coords = np.array(coords)
    if coords.ndim == 2:  # list of index pairs is passed
        return [convert_single(coord) for coord in coords]
    else:  # single index pair is passed
        return convert_single(coords)


# Converts voltage values to pixel indices
def convert_2_coords(volts, scan_params):  # assumes scan window is a square
    x_center, y_center, scan_range, steps = scan_params
    x_start = x_center - scan_range / 2
    y_start = y_center - scan_range / 2
    increment = scan_range / steps

    def convert_single(coord):
        vx, vy = coord
        x = (vx - x_start) / increment
        y = ((y_start + scan_range) - vy) / increment
        return [x, y]

    if isinstance(volts, list):
        volts = np.array(volts)
    if volts.ndim == 2:  # list of index pairs is passed
        return [convert_single(volt) for volt in volts]
    else:  # single index pair is passed
        return convert_single(volts)


def get_angle_parametrisation(image: np.ndarray):
    image_size = image.shape[0]

    # binarize image
    skel_image = skeletonize(image.astype(np.uint8))

    # identify triple points a priori
    triple_points = pcv.morphology.find_branch_pts(skel_img=skel_image.astype(np.uint8))

    # disconnect lines by subtracting their intersections
    dil_kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=[3, 3])
    triple_points_dilated = cv2.dilate(src=triple_points, kernel=dil_kernel, borderType=cv2.BORDER_ISOLATED, iterations=1)
    _, triple_points_dilated = cv2.threshold(triple_points_dilated, 0, 1, cv2.THRESH_BINARY)
    transition_lines = skel_image.astype(np.int8) - triple_points_dilated.astype(np.int8)
    transition_lines[transition_lines == -1] = 0

    # identify individual lines
    segment_objects, _ = cv2.findContours(image=transition_lines.astype(np.uint8), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    # Re-format found contours
    contours = []
    for contour in segment_objects:
        row_indices = contour[:, 0, 0]  # this is actually the wrong way around, but is corrected later
        col_indices = contour[:, 0, 1]
        contours.append((row_indices, col_indices))

    # Fit straight lines to found contours
    fit_coeffs = []
    for c in contours:
        coeffs = np.polyfit(c[0], c[1], deg=1, rcond=None, full=False, w=None, cov=False)
        fit_coeffs.append(coeffs)
    sorted_coeffs = sorted(fit_coeffs, key=lambda x: x[0])[::-1]

    # Order the line coefficients: vertical bottom, vertical top, horizontal left, horizontal right, diagonal (not used)
    y_values = [straight_line(image_size // 2, c[0], c[1]) for c in sorted_coeffs[:-1]]  # Calculate y-values for lead_transition lines at axis mid-point
    for i in range(0, 4, 2):  # iterate in pairs of vertical and horizontal transitions
        if y_values[i] < y_values[i + 1]:
            sorted_coeffs[i], sorted_coeffs[i + 1] = sorted_coeffs[i + 1], sorted_coeffs[i]  # Check and swap adjacent lines if necessary

    # Re-calculate triple points from intersection, and apex of meta-stable region
    intercept1_x, intercept1_y = two_line_intersection(sorted_coeffs[0][0], sorted_coeffs[2][0], sorted_coeffs[0][1], sorted_coeffs[2][1])  # triple left
    intercept2_x, intercept2_y = two_line_intersection(sorted_coeffs[1][0], sorted_coeffs[2][0], sorted_coeffs[1][1], sorted_coeffs[2][1])  # vertex bottom
    intercept3_x, intercept3_y = two_line_intersection(sorted_coeffs[1][0], sorted_coeffs[3][0], sorted_coeffs[1][1], sorted_coeffs[3][1])  # triple right
    intercept4_x, intercept4_y = two_line_intersection(sorted_coeffs[0][0], sorted_coeffs[3][0], sorted_coeffs[0][1], sorted_coeffs[3][1])  # vertex top

    # Find all angles for parametrizing the lines
    alpha1 = np.pi/2 - np.abs(np.arctan(sorted_coeffs[2][0]))  # horizontal left
    alpha2 = np.pi/2 - np.abs(np.arctan(sorted_coeffs[3][0]))  # horizontal right
    beta1 = np.abs(np.arctan(sorted_coeffs[0][0]))  # vertical bottom
    beta2 = np.abs(np.arctan(sorted_coeffs[1][0]))  # vertical top
    gamma1 = np.abs(np.arctan(two_point_slope(0, image_size, intercept1_x, intercept1_y)))  # bottom left corner
    gamma2 = np.abs(np.arctan(two_point_slope(image_size, image_size, intercept1_x, intercept1_y)))  # bottom right corner
    gamma3 = np.abs(np.arctan(two_point_slope(image_size, 0, intercept3_x, intercept3_y)))  # top right corner
    gamma4 = np.abs(np.arctan(two_point_slope(0, 0, intercept3_x, intercept3_y)))  # top left corner

    return [gamma1, gamma2, gamma3, gamma4, alpha1, alpha2, beta1, beta2]


## Backward reconstruction of image from angles to check they are correct
# Get triple points
def reconstruction_line(angles: list, image_size: int):
    gamma1, gamma2, gamma3, gamma4, alpha1, alpha2, beta1, beta2 = angles
    coeffs = np.zeros((8, 2))
    coeffs[0][0], coeffs[0][1] = -np.tan(gamma1), image_size  # bottom scalene left
    coeffs[1][0], coeffs[1][1] = np.tan(gamma2), get_c(image_size, image_size, np.tan(gamma2))  # bottom scalene right
    coeffs[2][0], coeffs[2][1] = -np.tan(gamma3), get_c(image_size, 0, -np.tan(gamma3))  # top scalene right
    coeffs[3][0], coeffs[3][1] = np.tan(gamma4), 0  # top scalene left
    left_triple_x, left_triple_y = two_line_intersection(coeffs[0][0], coeffs[1][0], coeffs[0][1], coeffs[1][1])
    right_triple_x, right_triple_y = two_line_intersection(coeffs[2][0], coeffs[3][0], coeffs[2][1], coeffs[3][1])
    left_triple = (left_triple_x, left_triple_y)
    right_triple = (right_triple_x, right_triple_y)

    # Get equations of lead transition lines
    coeffs[4][0], coeffs[4][1] = np.tan(np.pi/2 - alpha1), get_c(left_triple_x, left_triple_y, np.tan(np.pi/2 - alpha1))  # horizontal left
    coeffs[5][0], coeffs[5][1] = np.tan(beta1), get_c(left_triple_x, left_triple_y, np.tan(beta1))  # vertical bottom
    coeffs[6][0], coeffs[6][1] = np.tan(np.pi/2 - alpha2), get_c(right_triple_x, right_triple_y, np.tan(np.pi/2 - alpha2))  # horizontal right
    coeffs[7][0], coeffs[7][1] = np.tan(beta2), get_c(right_triple_x, right_triple_y, np.tan(beta2))  # vertical top

    return coeffs[4:], left_triple, right_triple


def construct_skeleton(lead_transition_coeffs, left_triple, right_triple, image_size):
    coeffs = lead_transition_coeffs
    # Get end coordinates of lead transition lines
    end1 = (0, int(coeffs[0][1]))  # left border
    end2 = (int((image_size - coeffs[1][1]) / coeffs[1][0]), image_size - 1)  # bottom border
    end3 = (image_size - 1, int(coeffs[2][0] * image_size + coeffs[2][1]))  # right border
    end4 = (int(-coeffs[3][1] / coeffs[3][0]), 0)  # top border

    # Construct array of line point indices, integers must be passed for it to work
    left_triple_int = tuple(map(round, left_triple))
    right_triple_int = tuple(map(round, right_triple))
    line1, line2 = bresenham_line(end1, left_triple_int), bresenham_line(left_triple_int, end2)
    line3, line4 = bresenham_line(end3, right_triple_int), bresenham_line(right_triple_int, end4)
    diag_line = bresenham_line(left_triple_int, right_triple_int)

    # Plot lines onto a 2D image
    image_boundary = np.zeros((image_size, image_size))
    for x, y in set(line1 + line2 + line3 + line4 + diag_line):
        image_boundary[y, x] = 1

    return image_boundary


def reconstruct_line_image(angles: list, image_size: int):
    line_coeffs, left_triple, right_triple = reconstruction_line(angles, image_size)
    skeleton = construct_skeleton(line_coeffs, left_triple, right_triple, image_size)
    return skeleton


def get_triple_points(image):
    # Add padding to the image
    padding = 2
    padded_image = np.pad(image, pad_width=padding, mode='edge')

    # binarize image
    skel_image = skeletonize(padded_image.astype(np.uint8))

    # identify triple points
    triple_points = pcv.morphology.find_branch_pts(skel_img=skel_image.astype(np.uint8))
    triple_coords = np.argwhere(triple_points == 255)

    triple_coords -= padding

    return triple_coords


def get_triangle_vertices(lead_transition_coeffs: list):
    coeffs = lead_transition_coeffs
    intercept1_x, intercept1_y = two_line_intersection(coeffs[0][0], coeffs[1][0], coeffs[0][1], coeffs[1][1])  # triple left
    intercept2_x, intercept2_y = two_line_intersection(coeffs[0][0], coeffs[3][0], coeffs[0][1], coeffs[3][1])  # vertex bottom
    intercept3_x, intercept3_y = two_line_intersection(coeffs[2][0], coeffs[3][0], coeffs[2][1], coeffs[3][1])  # triple right
    intercept4_x, intercept4_y = two_line_intersection(coeffs[1][0], coeffs[2][0], coeffs[1][1], coeffs[2][1])  # vertex top
    return [[intercept1_x, intercept1_y], [intercept4_x, intercept4_y], [intercept3_x, intercept3_y], [intercept2_x, intercept2_y]]


def get_triangle_pair(lead_transition_coeffs: list):
    coeffs = lead_transition_coeffs
    intercept1_x, intercept1_y = two_line_intersection(coeffs[0][0], coeffs[1][0], coeffs[0][1], coeffs[1][1])  # triple left
    intercept2_x, intercept2_y = two_line_intersection(coeffs[0][0], coeffs[3][0], coeffs[0][1], coeffs[3][1])  # vertex bottom
    intercept3_x, intercept3_y = two_line_intersection(coeffs[2][0], coeffs[3][0], coeffs[2][1], coeffs[3][1])  # triple right
    intercept4_x, intercept4_y = two_line_intersection(coeffs[1][0], coeffs[2][0], coeffs[1][1], coeffs[2][1])  # vertex top
    bottom_triangle = [[intercept2_x, intercept2_y], [intercept1_x, intercept1_y], [intercept3_x, intercept3_y]]
    top_triangle = [[intercept4_x, intercept4_y], [intercept1_x, intercept1_y], [intercept3_x, intercept3_y]]
    return bottom_triangle, top_triangle


def plot_triangles(lead_transition_coeffs: list):
    coeffs = lead_transition_coeffs
    intercept1_x, intercept1_y = two_line_intersection(coeffs[0][0], coeffs[1][0], coeffs[0][1], coeffs[1][1])  # triple left
    intercept2_x, intercept2_y = two_line_intersection(coeffs[0][0], coeffs[3][0], coeffs[0][1], coeffs[3][1])  # vertex bottom
    intercept3_x, intercept3_y = two_line_intersection(coeffs[2][0], coeffs[3][0], coeffs[2][1], coeffs[3][1])  # triple right
    intercept4_x, intercept4_y = two_line_intersection(coeffs[1][0], coeffs[2][0], coeffs[1][1], coeffs[2][1])  # vertex top
    x_points = [intercept1_x, intercept2_x, intercept3_x, intercept1_x, intercept4_x, intercept3_x]
    y_points = [intercept1_y, intercept2_y, intercept3_y, intercept1_y, intercept4_y, intercept3_y]
    return [x_points, y_points]


# Create a binary mask for a triangle given its vertices
def make_mask(image_size, vertices):
    mask = np.zeros((image_size, image_size), dtype=np.uint8)
    pts = np.array([vertices], dtype=np.int32)
    cv2.fillPoly(mask, pts, color=1)
    return mask


def make_metastable_masks(lead_transition_coeffs: list, image_size: int):
    coeffs = lead_transition_coeffs
    intercept1_x, intercept1_y = two_line_intersection(coeffs[0][0], coeffs[1][0], coeffs[0][1], coeffs[1][1])  # triple left
    intercept2_x, intercept2_y = two_line_intersection(coeffs[0][0], coeffs[3][0], coeffs[0][1], coeffs[3][1])  # vertex bottom
    intercept3_x, intercept3_y = two_line_intersection(coeffs[2][0], coeffs[3][0], coeffs[2][1], coeffs[3][1])  # triple right
    intercept4_x, intercept4_y = two_line_intersection(coeffs[1][0], coeffs[2][0], coeffs[1][1], coeffs[2][1])  # vertex top

    # Construct vertices enclosing meta-stable and stable regions
    vertices1 = [(intercept1_x, intercept1_y), (intercept2_x, intercept2_y), (intercept3_x, intercept3_y)]
    vertices2 = [(intercept1_x, intercept1_y), (intercept4_x, intercept4_y), (intercept3_x, intercept3_y)]

    # Finding and applying masks for the two possible meta-stable regions
    mask1 = make_mask(image_size, vertices1)  # bottom triangle (A)
    mask2 = make_mask(image_size, vertices2)  # top triangle (B)
    return mask1, mask2


# axis over the IDT, aligned with lead transitions, anchored at IDT midpoint
def get_detuning_axis(vertex_coords):
    x0, y0 = midpoint(vertex_coords[0][0], vertex_coords[0][1], vertex_coords[2][0], vertex_coords[2][1])
    slope = two_point_slope(vertex_coords[1][0], vertex_coords[1][1], vertex_coords[3][0], vertex_coords[3][1])
    return [x0, y0], slope


def get_sampling_points(metastable_vertices, n=4, contract=True, factor=0.5):

    def get_lengths(triangle_coords):
        vertex, left_triple, right_triple = triangle_coords

        length_left = two_point_distance(vertex, left_triple)
        length_right = two_point_distance(vertex, right_triple)
        length_base = two_point_distance(left_triple, right_triple)

        return np.array([length_left, length_right, length_base])

    def contract_triangle(triangle_coords, factor):
        centroid = np.mean(triangle_coords, axis=0)
        scaled_triangle = centroid + factor * (triangle_coords - centroid)
        return scaled_triangle

    def define_vectors(triangle_coords):
        vertex, left_triple, right_triple = triangle_coords

        vector_left = left_triple - vertex
        vector_right = right_triple - vertex
        vector_base = right_triple - left_triple

        mag_vl = np.linalg.norm(vector_left)
        mag_vr = np.linalg.norm(vector_right)
        mag_vb = np.linalg.norm(vector_base)

        unit_vector_left = vector_left / mag_vl
        unit_vector_right = vector_right / mag_vr
        unit_vector_base = vector_base / mag_vb

        return np.array([unit_vector_left, unit_vector_right, unit_vector_base])

    def create_internal_vertices(n, triangle_coords, side_lengths, basis_vectors):
        scaling_factor = 1 / n
        rescaled_lengths = side_lengths * scaling_factor
        vertex, _, _ = triangle_coords
        lr_basis_vec = basis_vectors[:-1] * rescaled_lengths[:-1][:, np.newaxis]

        interior_vertices = []
        index = 0
        for i in range(n + 1):
            for j in range(n + 1 - i):
                new_coord = vertex + np.dot(np.array([i, j]), lr_basis_vec)
                interior_vertices.append(new_coord)
                index += 1

        return np.array(interior_vertices)

    def generate_block(groups, start, skip_size):
        result = []
        for block_size in range(groups, 0, -1):
            block = list(range(start, start + block_size))
            result.extend(block)
            start += block_size + skip_size  # Move start index for the next block, skipping one index
        return np.array(result)

    def create_sampling_points(interior_vertices, n):
        num_first_sweep = sum(k for k in range(1, n + 2))
        p1_idx_first_sweep = generate_block(groups=n, start=0, skip_size=1)
        p2_idx_first_sweep = p1_idx_first_sweep + 1
        p3_idx_first_sweep = np.array(range(n + 1, num_first_sweep))

        p1_idx_second_sweep = generate_block(n - 1, start=n + 1, skip_size=1)
        p2_idx_second_sweep = p1_idx_second_sweep + 1
        p3_idx_second_sweep = generate_block(n - 1, start=1, skip_size=2)

        p1_indices = np.concatenate((p1_idx_first_sweep, p1_idx_second_sweep), axis=0)
        p2_indices = np.concatenate((p2_idx_first_sweep, p2_idx_second_sweep), axis=0)
        p3_indices = np.concatenate((p3_idx_first_sweep, p3_idx_second_sweep), axis=0)

        sampling_points = []
        for i, j, k in zip(p1_indices, p2_indices, p3_indices):
            p1 = interior_vertices[i]
            p2 = interior_vertices[j]
            p3 = interior_vertices[k]
            new_coord = np.mean([p1, p2, p3], axis=0)
            sampling_points.append(new_coord)

        return np.array(sampling_points)

    def run_sampler(triangle_coords):

        if contract:
            scaled_triangle = contract_triangle(triangle_coords, factor)
            triangle_coords = scaled_triangle

        lengths = get_lengths(triangle_coords)
        vectors = define_vectors(triangle_coords)

        internal_vertices = create_internal_vertices(n, triangle_coords, lengths, vectors)
        sampling_points = create_sampling_points(internal_vertices, n)

        if n > 2:  # adds vertex points of interior sub-triangles
            additional_points_idx = generate_block(n - 2, start=n + 2, skip_size=2)
            additional_points = internal_vertices[additional_points_idx]
            sampling_points = np.append(sampling_points, additional_points, axis=0)

        return sampling_points


    top_triangle = [metastable_vertices[1], metastable_vertices[0], metastable_vertices[2]]  # top vertex, left triple, right triple
    bottom_triangle = [metastable_vertices[3], metastable_vertices[0], metastable_vertices[2]]  # bottom vertex, left triple, right triple

    if n == 1:
        sampling_points_top = np.array([[np.mean(np.array(top_triangle)[:, 0]), np.mean(np.array(top_triangle)[:, 1])]]) 
        sampling_points_bottom = np.array([[np.mean(np.array(bottom_triangle)[:, 0]), np.mean(np.array(bottom_triangle)[:, 1])]])
    else:
        sampling_points_top = run_sampler(top_triangle)
        sampling_points_bottom = run_sampler(bottom_triangle)

    return sampling_points_top, sampling_points_bottom


def pulse_endpoints(vertex_coords, length):
    anchor, axis = get_detuning_axis(vertex_coords)
    peak_to_peak = two_point_distance(vertex_coords[1], vertex_coords[3])

    if length == 'auto':
        bottom_right_point = ray_along_direction([0, 0], axis, 1.3 * peak_to_peak)
        upper_left_point = ray_along_direction([0, 0], axis, -1.3 * peak_to_peak)

    else:
        assert 0.0 < length < 80.0, "Your chosen pulse length falls out of the bounds [0, 80] mV"
        bottom_right_point = ray_along_direction([0, 0], axis, length)
        upper_left_point = ray_along_direction([0, 0], axis, -length)
        
    return [bottom_right_point, upper_left_point]


def rotate_180_degrees(x, y, cx, cy):
    x_rotated = 2 * cx - x
    y_rotated = 2 * cy - y
    return x_rotated, y_rotated


def check_simplex_bounds(simplex, bounds):
    """
    Checks if the simplex vertices are within bounds and clips if necessary.

    Args:
    - simplex: np.array of shape (n_vertices, n_params), initial simplex vertices.
    - bounds: list of tuples [(lower1, upper1), (lower2, upper2), ...] defining parameter bounds.

    Returns:
    - corrected_simplex: np.array of the same shape as simplex, corrected to fit within bounds.
    """
    simplex_corrected = np.copy(simplex)

    for i, (lower, upper) in enumerate(bounds):
        # Check if each vertex violates the bounds and clip if necessary
        if np.any(simplex[:, i] < lower) or np.any(simplex[:, i] > upper):
            simplex_corrected[:, i] = np.clip(simplex[:, i], lower, upper)

    return simplex_corrected

