#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is file contains functions for plotting segmented images, cell boundaries,
histograms of cell properties, and analysis overlays.

Authors: 
    - Vansh Sharma, Michael Ullman and Venkat Raman

Affiliation: 
    - APCL Group 
    - Department of Aerospace Engineering, University of Michigan, Ann Arbor
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from astropy.stats import knuth_bin_width
from .utils import set_axes_style
 
def plot_segmentation(image, outlines, dpi, title="Cells Captured via Segmentation"):
    """
    Plots an image with segmentation outlines overlaid.

    Parameters:
    - image: 2D NumPy array, grayscale image.
    - outlines: List of outlines obtained from segmentation.
    - title: Title of the plot (default: "Cells Captured via Segmentation").
    """
    fig, ax = plt.subplots(dpi=dpi)  # Creates a figure and an axis
    ax.imshow(image, cmap='gray')  # Display the image
    for o in outlines:
        ax.plot(o[:, 0], o[:, 1], color='r', linewidth=0.75)  # Overlay outlines

    ax.set_title(title, fontsize=14, fontweight='bold')  # Set title
    ax.axis('off')  # Remove axis labels
    plt.tight_layout()
    plt.show()
    print("~~~ Plotted: segmented image ~~~")

def plot_distribution(data, title, xlabel, nbins, img_file_name, output_dir):
    """Plots histogram and kernel density estimation."""
    num_lines = len(data)
    output_path = output_dir+img_file_name
    fig, axes = plt.subplots(1, num_lines, figsize=(15, 6), dpi=350)

    for i in range(num_lines):
        ax = axes[i]
        set_axes_style(ax)
        
        if nbins is None:
            bin_width = knuth_bin_width(np.array(data[i]))
            nbins = np.arange(min(data[i]), max(data[i]) + bin_width, bin_width)

        ax.hist(data[i], bins=nbins, alpha=0.7, color='blue', density=True)

        if len(data[i]) > 1:
            kde = gaussian_kde(data[i])
            x_vals = np.linspace(min(data[i]), max(data[i]), 1000)
            kde_vals = kde(x_vals)
            ax.plot(x_vals, kde_vals, color='red', linewidth=2)

        ax.set_title(f'{title} - Line {i + 1}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Density')

    plt.tight_layout()
    plt.savefig(output_path, dpi=350, bbox_inches='tight', pad_inches=0, transparent=False)
    plt.show()
    print("~~~ Plotted & Saved: Statistics of the cells ~~~")

def plot_image_with_annotations(image, outlines, cell_centroids, results, line_positions, 
                                axisSel, dpi, num_lines, img_file_name, output_dir):
    """
    Plots an image with segmentation outlines, cell centroids, and analysis lines while preserving dimensions.

    Parameters:
    - image: 2D NumPy array, the grayscale image.
    - outlines: Segmentation outlines from ML model.
    - cell_centroids: Dictionary of cell centroids.
    - results: Dictionary containing cell data per line.
    - line_positions: List of positions for vertical/horizontal analysis lines.
    - axisSel: 'vertical' or 'horizontal' to indicate line orientation.
    - orginial_image_dpi: The original DPI of the image.
    - img_file_path: File path of the input image (used for saving output).
    - num_lines: Number of lines for analysis (default: 4).
    """

    # Ensure the figure size matches the image dimensions
    fig, ax = plt.subplots(figsize=(image.shape[1] / dpi, 
                                    image.shape[0] / dpi), 
                           dpi=dpi, frameon=False)
    
    # Plot the base image
    ax.imshow(image, cmap='gray')
    ax.axis('off')  # Turn off axis labels and ticks
    
    # Ensure the image maintains its original size
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)  # Invert y-axis for correct orientation

    # Prevent resizing or adjustments
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    output_path = output_dir+img_file_name
    # Save initial image without annotations
    # plt.savefig(output_path1, dpi=orginial_image_dpi, bbox_inches='tight', pad_inches=0, transparent=True)

    # Plot cell centroids
    for i in range(num_lines):
        cells_lines = results[i]
        for label in cells_lines:
            cell_centroid = cell_centroids[label[0]]
            ax.scatter(x=cell_centroid[1], y=cell_centroid[0], color='cyan', marker='o', s=50)

    # Overlay segmentation outlines
    for o in outlines:
        ax.plot(o[:, 0], o[:, 1], color='r', linewidth=0.75)

    # Plot analysis lines
    for pos in line_positions:
        if axisSel == 'vertical':
            ax.axvline(x=pos, color='yellow', linestyle='--', linewidth=2.2)
        elif axisSel == 'horizontal':
            ax.axhline(y=pos, color='yellow', linestyle='--', linewidth=2.2)

    # Ensure the image maintains its original size
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)  # Invert y-axis for correct orientation
    
    # Prevent resizing or adjustments
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save the figure with annotations
    # output_path = f'output_{img_file_path[-9:-4]}.png'
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.show()
    print("~~~ Plotted & Saved: Image with cells and annotations ~~~")
    
    
    
    