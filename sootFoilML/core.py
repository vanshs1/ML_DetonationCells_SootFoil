#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is file implements the main image segmentation pipeline. 
Includes preprocessing, denoising, and cell detection functions.

Authors: 
    - Vansh Sharma, Michael Ullman and Venkat Raman

Affiliation: 
    - APCL Group 
    - Department of Aerospace Engineering, University of Michigan, Ann Arbor
"""

import numpy as np
import cv2
from skimage.io import imread
from PIL import Image
from scipy.ndimage import center_of_mass
from cellpose import utils, models
from cellpose import denoise
from .config import (
    DEFAULT_MODEL_TYPE,
    DEFAULT_CHANNELS,
    DEFAULT_DIAMETER,
    FLOW_THRESHOLD,
    CELLPROB_THRESHOLD,
    NITER,
    RESAMPLE,
    DO_3D,
    USE_GPU,
    DENOISE_MODEL_TYPE
)

def load_image(file_path):
    """Loads an image from a given file path."""
    
    img = Image.open(file_path)
    orginial_image_dpi = img.info.get('dpi', (100, 100))[0]
    del img
    
    image_data = imread(file_path, as_gray=True)
    print("~~~ Image loaded ~~~")
    return np.asarray(image_data), orginial_image_dpi

def preprocess_image(image):
    """Preprocess the image by applying gradient-based edge enhancement."""
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3) ## to be used for Y major images
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    north_edges = -grad_y
    south_edges = grad_y
    
    north_edges[north_edges < 0] = 0
    south_edges[south_edges < 0] = 0
    
    north_edges = cv2.normalize(north_edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    south_edges = cv2.normalize(south_edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Create larger kernels for dilation to add more thickness towards specific directions
    # Using symmetrical kernels for centered thickening effect
    kernel_size = 5  # Adjust this to increase thickness
    north_kernel = np.ones((kernel_size, 1), np.uint8)
    south_kernel = np.ones((kernel_size, 1), np.uint8)
    
    # Apply dilation and erosion to add more thickness towards the desired directions while keeping edges centered
    south_thick_edges = cv2.dilate(south_edges, north_kernel, iterations=1)
    south_thick_edges = cv2.erode(south_thick_edges, north_kernel, iterations=1)
    
    north_thick_edges = cv2.dilate(north_edges, south_kernel, iterations=2)
    north_thick_edges = cv2.erode(north_thick_edges, south_kernel, iterations=1)
    
    # Combine North and South thickened edges without normalization to prevent blurriness
    combined_thick_edges = cv2.add(north_thick_edges, south_thick_edges)
    
    # Apply Laplacian filter for edge enhancement
    laplacian = cv2.Laplacian(combined_thick_edges, cv2.CV_64F)
    sharp_edges = cv2.convertScaleAbs(laplacian)  # Convert the result to unsigned 8-bit type
    
    # Normalize the final image to ensure it fits within the display range
    sharp_edges = cv2.normalize(sharp_edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    print("~~~ Image processed ~~~")
    return sharp_edges


def denoise_image(image):
    """
    Applies ML-based denoising to the image using CellPose DenoiseModel.

    Parameters:
    - image: 2D NumPy array of the input image.

    Returns:
    - Denoised image.
    """
    dn = denoise.DenoiseModel(model_type=DENOISE_MODEL_TYPE, gpu=USE_GPU)
    denoised_image = dn.eval(image, channels=None, diameter=None)[:, :, 0]
    print("~~~ Image denoised ~~~")
    return denoised_image


def segment_cells(
    image, 
    model_type=DEFAULT_MODEL_TYPE, 
    channels=DEFAULT_CHANNELS, 
    diameter=DEFAULT_DIAMETER, 
    use_gpu=USE_GPU
):
    """
    Runs segmentation on the given image.

    Parameters:
    - image: 2D NumPy array of the input image.
    - model_type: Type of model to use ('cyto', 'nuclei', etc.).
    - channels: List specifying which channels to use (e.g., [0,0] for grayscale).
    - diameter: Estimated cell diameter (None for automatic detection).
    - use_gpu: Whether to enable GPU acceleration (default from config.py).

    Additional parameters are loaded from the config file:
    - flow_threshold: Maximum allowed error for flow consistency.
    - cellprob_threshold: Threshold for cell probability.
    - niter: Number of iterations for convergence.
    - resample: Whether to resample the image.
    - do_3D: Enable 3D segmentation.

    Returns:
    - masks: Segmented masks.
    - flows: Flow dynamics.
    - styles: Style vector.
    - diams: Estimated diameters.
    """

    # Initialize model
    model = models.Cellpose(gpu=use_gpu, model_type=model_type)

    # Run segmentation with configurable parameters
    masks, flows, styles, diams = model.eval(
        image,
        diameter=diameter,
        channels=channels,
        flow_threshold=FLOW_THRESHOLD,
        cellprob_threshold=CELLPROB_THRESHOLD,
        niter=NITER,
        resample=RESAMPLE,
        do_3D=DO_3D
    )
    outlines = utils.outlines_list(masks)
    print("~~~ Image segmented ~~~")
    return masks, outlines, flows, styles, diams 

def extract_cell_properties(masks):
    """Extracts cell properties like area and centroid."""
    cell_areas = {}
    cell_centroids = {}

    for cell_label in np.unique(masks):
        if cell_label == 0:
            continue
        cell_areas[cell_label] = np.sum(masks == cell_label)
        cell_centroids[cell_label] = center_of_mass(masks == cell_label)
    print("~~~ Image cell properties extracted ~~~")
    return cell_areas, cell_centroids

def clip_image_and_masks(image, masks=None, clip_percent=0.025):
    """Clips a percentage of the image from each side."""
    height, width = image.shape[:2]
    clip_h, clip_w = int(height * clip_percent), int(width * clip_percent)
    
    clipped_image = image[clip_h:height - clip_h, clip_w:width - clip_w]
    clipped_masks = masks[clip_h:height - clip_h, clip_w:width - clip_w] if masks is not None else None
    
    return clipped_image, clipped_masks






