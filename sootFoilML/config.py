#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is file stores all configurable parameters such as segmentation settings, 
preprocessing options, model selection, and visualization preferences. 

Authors: 
    - Vansh Sharma, Michael Ullman and Venkat Raman

Affiliation: 
    - APCL Group 
    - Department of Aerospace Engineering, University of Michigan, Ann Arbor
"""


#### ~~~~~~~~~~~~ Pre-processing options
ENABLE_PREPROCESSING = False  # Toggle pre-processing enhancements
OVERLAY_COUNT = 8  # Number of times to overlay gradient images if pre-processing is enabled

#### ~~~~~~~~~~~~ Denoising via ML
ENABLE_DENOISING = False  # Toggle ML-based denoising
DENOISE_MODEL_TYPE = "denoise_cyto3"  # Denoising model type

#### ~~~~~~~~~~~~ Configuration for Soot Foil segmentation
DEFAULT_MODEL_TYPE = "cyto3"  # Options: 'cyto', 'nuclei', 'cyto2', etc.
DEFAULT_CHANNELS = [0, 0]  # Options: [0,0] (grayscale), [2,3] (G=cytoplasm, B=nucleus), etc.

#### ~~~~~~~~~~~~ Segmentation parameters
DEFAULT_DIAMETER = None  # Set diameter manually or use automatic estimation (None)
FLOW_THRESHOLD = 0.4  # Maximum allowed flow error for segmentation
CELLPROB_THRESHOLD = 0.0  # Threshold for cell probability
NITER = None  # Number of iterations (None means proportional to cell size)
RESAMPLE = True  # Whether to resample the image
DO_3D = False  # Enable 3D segmentation (option not developed)
USE_GPU = False  # Enable or disable GPU acceleration

#### ~~~~~~~~~~~~ Line positioning settings
NUM_LINES = 4          # Number of lines for analysis
AXIS_SELECTION = "vertical"  # Choose 'vertical' or 'horizontal'
NBINS = None            # Number of bins for histogram plots: (None) is for optimal calulation