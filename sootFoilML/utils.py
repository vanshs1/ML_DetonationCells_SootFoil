#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is file provides helper utilities for handling image transformations, 
calculating line positions, and processing detected cell properties.

Authors: 
    - Vansh Sharma, Michael Ullman and Venkat Raman

Affiliation: 
    - APCL Group 
    - Department of Aerospace Engineering, University of Michigan, Ann Arbor
"""

import numpy as np
import cv2

# Function to set the style for each axes object
def set_axes_style(ax):
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    # ax.set_yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=12, width=2, length=6, direction='in', top=True, bottom=True, left=True, right=True)
    ax.tick_params(axis='both', which='minor', width=1, length=3, direction='in', top=True, bottom=True, left=True, right=True)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(2)

def overlay_images(base_img, gradient_img, times=1, alpha=0.5):
    """Overlay gradient images to enhance features."""
    if base_img.shape != gradient_img.shape:
        raise ValueError("Base and gradient images must have the same shape")

    base_img = base_img.astype(np.float32)
    gradient_img = gradient_img.astype(np.float32)
    enhanced_img = base_img.copy()

    for _ in range(times):
        enhanced_img = cv2.addWeighted(enhanced_img, 1, gradient_img, alpha, 0)

    return np.clip(enhanced_img, 0, 255).astype(np.uint8)

def add_and_remove_padding(img, padding, color=(255, 255, 255), add=1):
    """Adds or removes padding from an image."""
    if add == 1:
        return cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=color)
    else:
        height, width = img.shape[:2]
        return img[padding:height - padding, padding:width - padding]


def calculate_line_positions(image_shape, num_lines=4, axisSel="vertical"):
    """
    Calculates equidistant line positions for image analysis.

    Parameters:
    - image_shape: Tuple (height, width) of the image.
    - num_lines: Number of lines to place.
    - axisSel: 'vertical' or 'horizontal' to determine line orientation.

    Returns:
    - line_positions: List of computed line positions.
    """
    height, width = image_shape[:2]

    if axisSel == "vertical":
        line_positions = np.linspace(0, width, num_lines + 2, dtype=int)[1:-1]  # Exclude the ends
    elif axisSel == "horizontal":
        line_positions = np.linspace(0, height, num_lines + 2, dtype=int)[1:-1]  # Exclude the ends
    else:
        raise ValueError("The 'axis' parameter should be 'vertical' or 'horizontal'.")
        
    return line_positions


def detect_cells_alongLines(masks, axis='vertical', line_positions=None):
    """
    Detects cells along vertical or horizontal lines on an image and plots the distributions 
    of Dx, Dy, and cell areas.

    Parameters:
    - masks: 2D array of the image to process, with cell labels.
    - axis: 'vertical' or 'horizontal' indicating the orientation of lines.
    - line_positions: List of custom positions for lines; if None, lines will be equally spaced.

    Returns:
    - results: A dictionary where each key is the line index and each value is a list of tuples 
      (cell_label, Dx, Dy, area) for cells intersected by that line.
    """
    height, width = masks.shape
    
    if line_positions is None:
        raise ValueError("Could not find line positions.")

    results = {i: [] for i in range(len(line_positions))}

    # Identify and analyze cells intersected by each line
    for cell_label in np.unique(masks):
        if cell_label == 0:  # Skip background
            continue

        mask = (masks == cell_label)
        coords = np.column_stack(np.where(mask))

        # Calculate Dx (horizontal span)
        min_coords = coords[np.argmin(coords[:, 1]), :]
        max_coords = coords[np.argmax(coords[:, 1]), :]
        Dx = np.sqrt((max_coords[1] - min_coords[1]) ** 2 + (max_coords[0] - min_coords[0]) ** 2)

        # Calculate Dy (vertical span)
        min_coords = coords[np.argmin(coords[:, 0]), :]
        max_coords = coords[np.argmax(coords[:, 0]), :]
        Dy = np.sqrt((max_coords[0] - min_coords[0]) ** 2 + (max_coords[1] - min_coords[1]) ** 2)

        cell_area = np.sum(mask)

        for i, line_position in enumerate(line_positions):
            if axis == 'vertical':
                min_y, min_x = coords.min(axis=0)
                max_y, max_x = coords.max(axis=0)
                if min_x <= line_position <= max_x:
                    results[i].append((cell_label, Dx, Dy, cell_area))
            elif axis == 'horizontal':
                min_y, min_x = coords.min(axis=0)
                max_y, max_x = coords.max(axis=0)
                if min_y <= line_position <= max_y:
                    results[i].append((cell_label, Dx, Dy, cell_area))

    print("~~~ Cells detected along the lines ~~~")
    return results



