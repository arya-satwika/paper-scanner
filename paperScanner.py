import argparse
import sys
from enum import IntEnum
from pathlib import Path

import cv2
import numpy as np


class OutputMode(IntEnum):
    COLOR = 0
    GRAY = 1
    THRESHOLD = 2


SHOW_STEPS_FILTER = frozenset({"original", "cropped", "final"})


class DebugViewer:
    def __init__(self, enabled=False, step_filter=None):
        self.enabled = enabled
        self.step_filter = step_filter

    def _should_show(self, step):
        if not self.enabled:
            return False

        if self.step_filter is None:
            return True

        return step in self.step_filter

    def show(self, title, image, step=None):
        if not self._should_show(step):
            return

        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.imshow(title, image)

        print(f"Showing step: {title}")
        print("Press any key to continue, or ESC to continue anyway.")

        cv2.waitKey(0)

    def show_points(self, title, image, points, step=None):
        if not self._should_show(step):
            return

        preview = image.copy()

        if len(preview.shape) == 2:
            preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)

        points = np.asarray(points, dtype=np.float32)

        for index, point in enumerate(points):
            x, y = int(point[0]), int(point[1])

            cv2.circle(
                preview,
                (x, y),
                8,
                (0, 255, 0),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

            cv2.putText(
                preview,
                str(index + 1),
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        ordered = order_points(points)

        for index in range(4):
            point_a = tuple(ordered[index].astype(int))
            point_b = tuple(ordered[(index + 1) % 4].astype(int))

            cv2.line(
                preview,
                point_a,
                point_b,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

        self.show(title, preview, step=step)

def resize_for_detection(image, max_width=1280):
    """
    Resize the image only for document detection.

    Returns:
        resized_image: smaller image used for edge/contour detection
        scale_back: multiply detected points by this to map back to original image
    """
    height, width = image.shape[:2]

    if width <= max_width:
        return image.copy(), 1.0

    scale = max_width / width
    new_width = max_width
    new_height = int(height * scale)

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    scale_back = width / new_width
    return resized, scale_back


def order_points(points):
    """
    Order 4 points as:

        top-left, top-right, bottom-right, bottom-left

    OpenCV image coordinates have:
        x increasing to the right
        y increasing downward
    """
    points = np.asarray(points, dtype=np.float32)

    center = points.mean(axis=0)

    angles = np.arctan2(
        points[:, 1] - center[1],
        points[:, 0] - center[0],
    )

    ordered = points[np.argsort(angles)]

    top_left_index = np.argmin(ordered[:, 0] + ordered[:, 1])
    ordered = np.roll(ordered, -top_left_index, axis=0)

    return ordered.astype(np.float32)


def distance(point_a, point_b):
    return float(np.linalg.norm(point_a - point_b))


def four_point_warp(image, points):
    """
    Apply a perspective transform so the selected 4-point quadrilateral
    becomes a flat rectangle.
    """
    rect = order_points(points)

    top_left, top_right, bottom_right, bottom_left = rect

    width_top = distance(top_right, top_left)
    width_bottom = distance(bottom_right, bottom_left)
    output_width = int(max(width_top, width_bottom))

    height_right = distance(bottom_right, top_right)
    height_left = distance(bottom_left, top_left)
    output_height = int(max(height_right, height_left))

    output_width = max(output_width, 1)
    output_height = max(output_height, 1)

    destination = np.array(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ],
        dtype=np.float32,
    )

    transform = cv2.getPerspectiveTransform(rect, destination)

    warped = cv2.warpPerspective(
        image,
        transform,
        (output_width, output_height),
    )

    return warped


def auto_canny_thresholds(image, sigma=0.5):
    median = float(np.median(image))

    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))

    if upper <= lower:
        upper = min(255, lower + 1)

    return lower, upper


def clamp_corners(corners, width, height):
    corners = np.asarray(corners, dtype=np.float32).copy()
    corners[:, 0] = np.clip(corners[:, 0], 0, width - 1)
    corners[:, 1] = np.clip(corners[:, 1], 0, height - 1)
    return corners


def score_document_quad(corners, image_shape):
    """
    Rank a quadrilateral as a document candidate.

    Higher scores are better. Negative means reject.
    """
    height, width = image_shape[:2]
    image_area = height * width

    corners = np.asarray(corners, dtype=np.float32)

    if (
        np.any(corners[:, 0] < -width * 0.08)
        or np.any(corners[:, 1] < -height * 0.08)
        or np.any(corners[:, 0] > width * 1.08)
        or np.any(corners[:, 1] > height * 1.08)
    ):
        return -1.0

    corners = clamp_corners(corners, width, height)
    area = cv2.contourArea(corners.astype(np.int32))
    area_ratio = area / image_area

    if area_ratio < 0.12 or area_ratio > 0.92:
        return -1.0

    ordered = order_points(corners)
    top_left, top_right, bottom_right, bottom_left = ordered

    margin_x = width * 0.02
    margin_y = height * 0.02
    edge_corners = sum(
        1
        for point in ordered
        if point[0] <= margin_x
        or point[0] >= width - margin_x - 1
        or point[1] <= margin_y
        or point[1] >= height - margin_y - 1
    )

    doc_width = max(
        distance(top_right, top_left),
        distance(bottom_right, bottom_left),
    )
    doc_height = max(
        distance(bottom_right, top_right),
        distance(bottom_left, top_left),
    )

    if doc_width < 1.0 or doc_height < 1.0:
        return -1.0

    if (
        edge_corners >= 4
        and area_ratio > 0.85
        and doc_width > width * 0.94
        and doc_height > height * 0.94
    ):
        return -1.0

    aspect = doc_width / doc_height
    if aspect < 0.25 or aspect > 4.0:
        return -1.0

    rotated_rect = cv2.minAreaRect(corners)
    rect_area = max(rotated_rect[1][0] * rotated_rect[1][1], 1.0)
    rectangularity = area / rect_area

    if rectangularity < 0.72:
        return -1.0

    edge_penalty = 0.9**edge_corners
    return area_ratio * rectangularity * edge_penalty


def close_edge_map(edges, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)


def build_canny_edges(blurred):
    lower_threshold, upper_threshold = auto_canny_thresholds(blurred)
    return cv2.Canny(blurred, lower_threshold, upper_threshold)


def build_morph_gradient_edges(blurred):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    gradient = cv2.morphologyEx(blurred, cv2.MORPH_GRADIENT, kernel)
    return cv2.Canny(gradient, 30, 100)


def build_clahe_edges(gray):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    return build_canny_edges(blurred)


def clean_paper_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)


def largest_connected_component(mask):
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8),
        connectivity=8,
    )

    if component_count < 2:
        return None

    largest_index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = np.zeros_like(mask)
    component[labels == largest_index] = 255
    return component


def build_otsu_paper_mask(blurred):
    _, mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    white_mean = float(blurred[mask == 255].mean()) if np.any(mask == 255) else 0.0
    black_mean = float(blurred[mask == 0].mean()) if np.any(mask == 0) else 255.0

    if white_mean < black_mean:
        mask = cv2.bitwise_not(mask)

    return clean_paper_mask(mask)


def build_percentile_paper_mask(blurred, percentile):
    threshold = float(np.percentile(blurred, percentile))
    mask = (blurred >= threshold).astype(np.uint8) * 255
    return clean_paper_mask(mask)


def quad_from_contours(contours, image_shape, min_area_ratio=0.12):
    """
    Find the best 4-corner document candidate among contours.
    """
    image_area = image_shape[0] * image_shape[1]
    min_document_area = image_area * min_area_ratio

    best_corners = None
    best_score = -1.0
    best_contour = None
    best_hull_area = 0.0

    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)

        if area < min_document_area:
            continue

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        perimeter = cv2.arcLength(hull, True)

        for epsilon_ratio in (0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05):
            approx = cv2.approxPolyDP(
                hull,
                epsilon_ratio * perimeter,
                True,
            )

            if len(approx) != 4:
                continue

            corners = approx.reshape(4, 2).astype(np.float32)
            score = score_document_quad(corners, image_shape)

            if score > best_score:
                best_score = score
                best_corners = clamp_corners(
                    corners,
                    image_shape[1],
                    image_shape[0],
                )

        if best_corners is None and hull_area > best_hull_area:
            best_hull_area = hull_area
            best_contour = hull

    return best_corners, best_contour


def _line_from_points(point_a, point_b):
    x1, y1 = float(point_a[0]), float(point_a[1])
    x2, y2 = float(point_b[0]), float(point_b[1])
    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1
    return a, b, c


def _intersect_lines(line_a, line_b):
    a1, b1, c1 = line_a
    a2, b2, c2 = line_b
    determinant = a1 * b2 - a2 * b1

    if abs(determinant) < 1e-6:
        return None

    return np.array(
        [
            (b1 * c2 - b2 * c1) / determinant,
            (c1 * a2 - c2 * a1) / determinant,
        ],
        dtype=np.float32,
    )


def _fit_line_from_points(points):
    points = np.asarray(points, dtype=np.float32)

    if len(points) < 2:
        return None

    line = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    vx = float(line[0])
    vy = float(line[1])
    x0 = float(line[2])
    y0 = float(line[3])
    point_a = np.array([x0, y0], dtype=np.float32)
    point_b = point_a + np.array([vx, vy], dtype=np.float32) * 1000.0
    return _line_from_points(point_a, point_b)


def _largest_bright_segment(values, min_brightness, min_fraction=0.12):
    bright = values >= min_brightness

    if bright.mean() < min_fraction:
        return None

    indices = np.where(bright)[0]

    if len(indices) == 0:
        return None

    splits = np.where(np.diff(indices) > 1)[0]
    runs = []
    start = 0

    for split_index in splits:
        runs.append((indices[start], indices[split_index]))
        start = split_index + 1

    runs.append((indices[start], indices[-1]))
    return max(runs, key=lambda run: run[1] - run[0])


def _filter_side_points(points, axis_index, remove_high_outliers=False):
    points = np.asarray(points, dtype=np.float32)

    if len(points) < 8:
        return points

    values = points[:, axis_index]
    lower_quartile = np.percentile(values, 25)
    upper_quartile = np.percentile(values, 75)
    spread = max(upper_quartile - lower_quartile, 1.0)

    if remove_high_outliers:
        cutoff = upper_quartile + spread * 0.8
        return points[values <= cutoff]

    cutoff = lower_quartile - spread * 0.8
    return points[values >= cutoff]


def _paper_edges_on_row(saturation_row, value_row, width, gap_threshold=40):
    paper_columns = np.where((saturation_row < 85) & (value_row > 95))[0]

    if len(paper_columns) < width * 0.15:
        return None

    column_gaps = np.diff(paper_columns)

    if len(column_gaps) == 0:
        return int(paper_columns[0]), int(paper_columns[-1])

    if column_gaps.max() > gap_threshold:
        gap_index = int(np.argmax(column_gaps))
        return int(paper_columns[0]), int(paper_columns[gap_index])

    return int(paper_columns[0]), int(paper_columns[-1])


def _paper_edges_on_column(saturation_column, value_column, height, gap_threshold=40):
    paper_rows = np.where((saturation_column < 85) & (value_column > 95))[0]

    if len(paper_rows) < height * 0.15:
        return None

    row_gaps = np.diff(paper_rows)

    if len(row_gaps) == 0:
        return int(paper_rows[0]), int(paper_rows[-1])

    if row_gaps.max() > gap_threshold:
        gap_index = int(np.argmax(row_gaps))
        return int(paper_rows[0]), int(paper_rows[gap_index])

    return int(paper_rows[0]), int(paper_rows[-1])


def find_corners_scanline(image):
    """
    Estimate document edges by scanning rows/columns for paper pixels.
    Uses HSV on color images so dark objects (e.g. keyboards) are excluded.
    """
    height, width = image.shape[:2]
    left_points = []
    right_points = []
    top_points = []
    bottom_points = []

    if len(image.shape) == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        for y in range(int(height * 0.05), int(height * 0.95), 2):
            edges = _paper_edges_on_row(saturation[y], value[y], width)

            if edges is None:
                continue

            left_x, right_x = edges

            if right_x - left_x < width * 0.28:
                continue

            left_points.append([left_x, y])
            right_points.append([right_x, y])

        for x in range(int(width * 0.05), int(width * 0.95), 2):
            edges = _paper_edges_on_column(saturation[:, x], value[:, x], height)

            if edges is None:
                continue

            top_y, bottom_y = edges

            if bottom_y - top_y < height * 0.22:
                continue

            top_points.append([x, top_y])
            bottom_points.append([x, bottom_y])
    else:
        brightness_threshold = float(np.percentile(image, 38))

        for y in range(0, height, 2):
            segment = _largest_bright_segment(image[y], brightness_threshold, 0.1)

            if segment is None:
                continue

            left_x, right_x = segment

            if right_x - left_x < width * 0.28:
                continue

            left_points.append([left_x, y])
            right_points.append([right_x, y])

        for x in range(0, width, 2):
            segment = _largest_bright_segment(image[:, x], brightness_threshold, 0.1)

            if segment is None:
                continue

            top_y, bottom_y = segment

            if bottom_y - top_y < height * 0.22:
                continue

            top_points.append([x, top_y])
            bottom_points.append([x, bottom_y])

    if min(len(left_points), len(right_points), len(top_points), len(bottom_points)) < 5:
        return None

    right_points = _filter_side_points(right_points, 0, remove_high_outliers=True)
    left_points = _filter_side_points(left_points, 0, remove_high_outliers=False)

    left_line = _fit_line_from_points(left_points)
    right_line = _fit_line_from_points(right_points)
    top_line = _fit_line_from_points(top_points)
    bottom_line = _fit_line_from_points(bottom_points)

    if None in (left_line, right_line, top_line, bottom_line):
        return None

    top_left = _intersect_lines(top_line, left_line)
    top_right = _intersect_lines(top_line, right_line)
    bottom_right = _intersect_lines(bottom_line, right_line)
    bottom_left = _intersect_lines(bottom_line, left_line)

    left_array = np.asarray(left_points, dtype=np.float32)
    right_array = np.asarray(right_points, dtype=np.float32)
    top_array = np.asarray(top_points, dtype=np.float32)
    bottom_array = np.asarray(bottom_points, dtype=np.float32)

    if any(point is None for point in (top_left, top_right, bottom_right, bottom_left)):
        corners = np.array(
            [
                [np.percentile(left_array[:, 0], 12), np.percentile(top_array[:, 1], 12)],
                [np.percentile(right_array[:, 0], 35), np.percentile(top_array[:, 1], 12)],
                [np.percentile(right_array[:, 0], 35), np.percentile(bottom_array[:, 1], 88)],
                [np.percentile(left_array[:, 0], 12), np.percentile(bottom_array[:, 1], 88)],
            ],
            dtype=np.float32,
        )
    else:
        corners = np.array(
            [top_left, top_right, bottom_right, bottom_left],
            dtype=np.float32,
        )

        top_y = float(np.percentile(top_array[:, 1], 8))
        bottom_y = float(np.percentile(bottom_array[:, 1], 92))
        left_x = float(np.percentile(left_array[:, 0], 12))
        right_x = float(np.percentile(right_array[:, 0], 35))

        corners[0] = [max(corners[0][0], left_x), min(corners[0][1], top_y)]
        corners[1] = [min(corners[1][0], right_x), min(corners[1][1], top_y)]
        corners[2] = [min(corners[2][0], right_x), max(corners[2][1], bottom_y)]
        corners[3] = [max(corners[3][0], left_x), max(corners[3][1], bottom_y)]

    return clamp_corners(corners, width, height)


def build_reference_paper_mask(blurred, color_image):
    height, width = blurred.shape
    combined = np.zeros((height, width), dtype=np.uint8)

    masks = [
        build_otsu_paper_mask(blurred),
        build_percentile_paper_mask(blurred, 42),
        build_percentile_paper_mask(blurred, 52),
    ]

    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    hsv_mask = ((saturation < 75) & (value > 85)).astype(np.uint8) * 255
    masks.append(clean_paper_mask(hsv_mask))

    for mask in masks:
        combined = cv2.bitwise_or(combined, mask)

    return largest_connected_component(combined)


def mask_coverage_score(corners, paper_mask):
    if paper_mask is None:
        return 0.0

    height, width = paper_mask.shape[:2]
    ordered = order_points(corners).astype(np.int32)

    quad_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(quad_mask, ordered, 255)

    paper_pixels = paper_mask > 0
    quad_pixels = quad_mask > 0
    paper_count = int(np.count_nonzero(paper_pixels))

    if paper_count == 0:
        return 0.0

    overlap = int(np.count_nonzero(paper_pixels & quad_pixels))
    recall = overlap / paper_count
    precision = overlap / max(int(np.count_nonzero(quad_pixels)), 1)
    return (2.0 * recall * precision) / max(recall + precision, 1e-6)


def _touches_image_border(corners, image_shape, axis="x", side="high", threshold=0.97):
    width = image_shape[1]
    height = image_shape[0]
    ordered = order_points(corners)

    if axis == "x" and side == "high":
        return ordered[1][0] >= width * threshold or ordered[2][0] >= width * threshold

    if axis == "x" and side == "low":
        return ordered[0][0] <= width * (1.0 - threshold) or ordered[3][0] <= width * (1.0 - threshold)

    if axis == "y" and side == "high":
        return ordered[2][1] >= height * threshold or ordered[3][1] >= height * threshold

    return ordered[0][1] <= height * (1.0 - threshold) or ordered[1][1] <= height * (1.0 - threshold)


def score_quad_candidate(corners, image_shape, paper_mask=None):
    geometry_score = score_document_quad(corners, image_shape)

    if geometry_score < 0.0:
        return -1.0

    coverage = mask_coverage_score(corners, paper_mask)
    score = geometry_score * (0.45 + 0.55 * coverage)

    if _touches_image_border(corners, image_shape, axis="x", side="high"):
        score *= 0.55

    return score


def _consider_quad_candidate(corners, image_shape, paper_mask, best_corners, best_score):
    if corners is None:
        return best_corners, best_score

    score = score_quad_candidate(corners, image_shape, paper_mask)

    if score > best_score:
        return corners, score

    return best_corners, best_score


def quad_from_paper_mask(mask, image_shape):
    component = largest_connected_component(mask)

    if component is None:
        return None

    contours, _ = cv2.findContours(
        component,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None

    corners, _ = quad_from_contours(contours, image_shape)
    return corners


def find_corners_from_contour_extremes(contour, image_shape):
    points = contour.reshape(-1, 2).astype(np.float32)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).ravel()

    corners = np.array(
        [
            points[np.argmin(sums)],
            points[np.argmin(diffs)],
            points[np.argmax(sums)],
            points[np.argmax(diffs)],
        ],
        dtype=np.float32,
    )

    return clamp_corners(corners, image_shape[1], image_shape[0])


def find_document_corners(image, viewer=None):
    """
    Try to find the document's four corners.

    Combines scanline boundary detection, edge finding, and bright-paper masks.
    Candidates are ranked by geometry and overlap with a reference paper mask.
    """
    if viewer is None:
        viewer = DebugViewer(False)

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    paper_mask = build_reference_paper_mask(blurred, image)

    best_corners = None
    best_score = -1.0
    best_contour = None
    best_hull_area = 0.0

    scanline_corners = find_corners_scanline(image)

    if scanline_corners is not None:
        scanline_score = score_quad_candidate(
            scanline_corners,
            image.shape,
            paper_mask,
        )

        if scanline_score > best_score:
            best_corners = scanline_corners
            best_score = scanline_score

    edge_pipelines = [
        build_canny_edges(blurred),
        build_morph_gradient_edges(blurred),
        build_clahe_edges(gray),
    ]

    for edges in edge_pipelines:
        closed_edges = close_edge_map(edges)

        contours, _ = cv2.findContours(
            closed_edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            continue

        corners, fallback_contour = quad_from_contours(contours, image.shape)
        best_corners, best_score = _consider_quad_candidate(
            corners,
            image.shape,
            paper_mask,
            best_corners,
            best_score,
        )

        if fallback_contour is not None:
            hull_area = cv2.contourArea(fallback_contour)

            if hull_area > best_hull_area:
                best_hull_area = hull_area
                best_contour = fallback_contour

    paper_masks = [
        build_otsu_paper_mask(blurred),
        build_percentile_paper_mask(blurred, 44),
        build_percentile_paper_mask(blurred, 50),
        build_percentile_paper_mask(blurred, 56),
    ]

    for mask in paper_masks:
        component = largest_connected_component(mask)

        if component is None:
            continue

        contours, _ = cv2.findContours(
            component,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )

        if not contours:
            continue

        largest_contour = max(contours, key=cv2.contourArea)
        corners, _ = quad_from_contours(contours, image.shape)
        best_corners, best_score = _consider_quad_candidate(
            corners,
            image.shape,
            paper_mask,
            best_corners,
            best_score,
        )

        extreme_corners = find_corners_from_contour_extremes(
            largest_contour,
            image.shape,
        )
        best_corners, best_score = _consider_quad_candidate(
            extreme_corners,
            image.shape,
            paper_mask,
            best_corners,
            best_score,
        )

    if (
        scanline_corners is not None
        and best_corners is not None
        and _touches_image_border(best_corners, image.shape, axis="x", side="high")
        and not _touches_image_border(scanline_corners, image.shape, axis="x", side="high")
    ):
        best_corners = scanline_corners

    if best_corners is not None:
        return clamp_corners(best_corners, width, height)

    if best_contour is None:
        return None

    rotated_rect = cv2.minAreaRect(best_contour)
    box = cv2.boxPoints(rotated_rect)
    return clamp_corners(box.astype(np.float32), width, height)


def brighten_image(image, amount=35):
    """
    Add brightness to the image.

    Formula:
        output = image * alpha + beta

    Here:
        alpha = 1.0
        beta = amount
    """
    return cv2.convertScaleAbs(image, alpha=1.0, beta=amount)


def make_threshold_scan(gray_image, viewer=None):
    """
    Produce a black-text-on-white-background scan.

    Combines Otsu and a mild adaptive threshold so thin strokes are not
    eaten by an overly aggressive local threshold (large block / high C).
    """
    if viewer is None:
        viewer = DebugViewer(False)

    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    a, otsu = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        4,
    )

    # White only where both methods agree — keeps faint/thin text intact.
    thresholded = cv2.bitwise_and(otsu, adaptive)

    # Reconnect tiny breaks in glyphs without noticeably bolding text.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresholded = cv2.morphologyEx(
        thresholded,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1,
    )

    viewer.show("3. Final scan", thresholded, step="final")

    return thresholded

def render_mode(color_image, mode, viewer=None):
    if viewer is None:
        viewer = DebugViewer(False)

    brightened = brighten_image(color_image)

    if mode == OutputMode.COLOR:
        return brightened

    gray = cv2.cvtColor(brightened, cv2.COLOR_BGR2GRAY)

    if mode == OutputMode.GRAY:
        return gray

    return make_threshold_scan(gray, viewer)

def capture_from_camera(camera_index):
    capture = cv2.VideoCapture(camera_index)

    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")

    print("Press SPACE to capture a frame.")
    print("Press ESC to cancel.")

    while True:
        success, frame = capture.read()

        if not success or frame is None:
            capture.release()
            raise RuntimeError("Could not read frame from camera")

        cv2.imshow("Webcam preview", frame)

        key = cv2.waitKey(30) & 0xFF

        if key == 27:
            capture.release()
            cv2.destroyWindow("Webcam preview")
            return None

        if key == 32:
            capture.release()
            cv2.destroyWindow("Webcam preview")
            return frame.copy()


class ScannerApp:
    def __init__(self, warped_color, viewer=None):
        self.main_window = "Scanned image"
        self.controls_window = "Scanner controls"
        self.crop_window = "Crop window"

        self.working_color = warped_color.copy()
        self.current_image = None

        self.mode = OutputMode.THRESHOLD

        self.crop_points = []
        self.is_cropping = False

        self.viewer = viewer or DebugViewer(False)
        self.has_shown_render_steps = False

    def run(self):
        cv2.namedWindow(self.main_window, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.controls_window, cv2.WINDOW_AUTOSIZE)

        cv2.createTrackbar(
            "Mode: 0 Color | 1 Gray | 2 Threshold",
            self.controls_window,
            int(self.mode),
            2,
            self.on_mode_change,
        )

        self.render()

        print()
        print("Controls:")
        print("  r   rotate right")
        print("  l   rotate left")
        print("  c   crop with four clicks")
        print("  s   save as Scanned.jpg")
        print("  Esc exit")
        print()

        while True:
            key = cv2.waitKey(30) & 0xFF

            if key == 27:
                break

            if key in (ord("r"), ord("R")):
                self.rotate_right()

            elif key in (ord("l"), ord("L")):
                self.rotate_left()

            elif key in (ord("c"), ord("C")):
                self.start_crop()

            elif key in (ord("s"), ord("S")):
                self.save_current_image()

        cv2.destroyAllWindows()

    def render(self):
        debug_viewer = None

        if not self.has_shown_render_steps:
            debug_viewer = self.viewer
            self.has_shown_render_steps = True

        self.current_image = render_mode(
            self.working_color,
            self.mode,
            debug_viewer,
        )

        cv2.imshow(self.main_window, self.current_image)

    def on_mode_change(self, value):
        self.mode = OutputMode(value)
        self.render()

    def rotate_right(self):
        self.working_color = cv2.rotate(
            self.working_color,
            cv2.ROTATE_90_CLOCKWISE,
        )
        self.render()

    def rotate_left(self):
        self.working_color = cv2.rotate(
            self.working_color,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
        )
        self.render()

    def start_crop(self):
        self.is_cropping = True
        self.crop_points = []

        cv2.namedWindow(self.crop_window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.crop_window, self.on_crop_mouse)

        self.draw_crop_preview()

        print("Crop mode: click four corners of the region you want to keep.")

    def on_crop_mouse(self, event, x, y, flags, userdata):
        if not self.is_cropping:
            return

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if len(self.crop_points) >= 4:
            return

        self.crop_points.append([float(x), float(y)])
        self.draw_crop_preview()

        if len(self.crop_points) == 4:
            self.finish_crop()

    def draw_crop_preview(self):
        preview = self.current_image.copy()

        if len(preview.shape) == 2:
            preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)

        for index, point in enumerate(self.crop_points):
            x, y = int(point[0]), int(point[1])

            cv2.circle(
                preview,
                (x, y),
                6,
                (0, 255, 0),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

            cv2.putText(
                preview,
                str(index + 1),
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        for index in range(1, len(self.crop_points)):
            previous_point = tuple(map(int, self.crop_points[index - 1]))
            current_point = tuple(map(int, self.crop_points[index]))

            cv2.line(
                preview,
                previous_point,
                current_point,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(self.crop_window, preview)

    def finish_crop(self):
        points = np.array(self.crop_points, dtype=np.float32)

        self.working_color = four_point_warp(self.working_color, points)

        self.is_cropping = False
        self.crop_points = []

        cv2.destroyWindow(self.crop_window)

        self.has_shown_render_steps = False
        self.render()

    def save_current_image(self):
        output_path = "Scanned.jpg"

        success = cv2.imwrite(output_path, self.current_image)

        if not success:
            print(f"Could not save {output_path}")
            return

        print(f"Saved {output_path}")


def load_source_image(args):
    if args.camera is not None:
        return capture_from_camera(args.camera)

    if args.image is None:
        raise RuntimeError("You must provide an image path or use --camera")

    image_path = Path(args.image)

    if not image_path.exists():
        raise RuntimeError(f"Image does not exist: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise RuntimeError(f"Could not open image: {image_path}")

    return image


def scan_document(source, viewer=None):
    if viewer is None:
        viewer = DebugViewer(False)

    viewer.show("1. Original image", source, step="original")

    detection_image, scale_back = resize_for_detection(source)

    corners = find_document_corners(detection_image, viewer)

    if corners is None:
        raise RuntimeError("Could not detect document corners")

    full_size_corners = corners * scale_back
    warped = four_point_warp(source, full_size_corners)

    viewer.show("2. Cropped document", warped, step="cropped")

    return warped

def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple OpenCV paper scanner",
    )

    parser.add_argument(
        "image",
        nargs="?",
        help="Path to an input image",
    )

    parser.add_argument(
        "--camera",
        nargs="?",
        const=0,
        type=int,
        help="Capture from webcam. Optionally provide camera index.",
    )

    parser.add_argument(
        "--show-steps",
        action="store_true",
        help="Show original, cropped, and final scan (press a key to advance each).",
    )

    return parser.parse_args()

def main():
    args = parse_args()

    try:
        source = load_source_image(args)

        if source is None:
            print("Capture cancelled.")
            return 1

        step_filter = SHOW_STEPS_FILTER if args.show_steps else None
        viewer = DebugViewer(args.show_steps, step_filter=step_filter)

        warped = scan_document(source, viewer)

        app = ScannerApp(warped, viewer)
        app.run()

        return 0

    except RuntimeError as error:
        print(error, file=sys.stderr)
        cv2.destroyAllWindows()
        return 1
    
if __name__ == "__main__":
    raise SystemExit(main())
