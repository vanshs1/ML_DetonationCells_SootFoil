#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is an example code to be included as supplementary material to the following article: 
"A Machine Learning Based Approach for Statistical Analysis of Detonation Cells from Soot Foils".

This is a demonstration intended to provide a working example for detecting, segmenting,  
and extracting statistics of cells in a sootfoil pattern. 
The example data is provided in this repo.
For application on other datasets, the requirement is to configure the settings in config.py. 

Dependencies: 
    - Python package list provided: package.list

Running the code: 
    - assuming above dependencies are configured, "python main_sootFoilSeg.py" will run the demo code. 
    NOTE - It is important to check config.py prior to running this code.

Authors: 
    - Vansh Sharma, Michael Ullman and Venkat Raman

Affiliation: 
    - APCL Group 
    - Department of Aerospace Engineering, University of Michigan, Ann Arbor
"""

import os
import json

from sootFoilML.core import (
    load_image, 
    preprocess_image, 
    segment_cells, 
    extract_cell_properties,
    denoise_image
)

from sootFoilML.visualization import ( 
    plot_segmentation, 
    plot_distribution,
    plot_image_with_annotations
)

from sootFoilML.utils import ( 
    overlay_images, 
    calculate_line_positions, 
    detect_cells_alongLines
)

from sootFoilML.config import (
    ENABLE_PREPROCESSING,
    ENABLE_DENOISING,
    OVERLAY_COUNT,
    DEFAULT_MODEL_TYPE,
    DEFAULT_CHANNELS,
    USE_GPU,
    NUM_LINES,
    AXIS_SELECTION,
    NBINS
)

def main():
    """Main execution script for analyzing images."""
    input_dir = "./example/"
    image_name = "case_10a.png"
    output_dir = "output/"
    os.makedirs(output_dir, exist_ok=True)

    img_file = os.path.join(input_dir, image_name)
    image, orginial_image_dpi = load_image(img_file)
    
    # Apply ML-based denoising if enabled
    if ENABLE_DENOISING:
        image = denoise_image(image)

    # Apply pre-processing enhancements if enabled: This should be user customized
    if ENABLE_PREPROCESSING:
        sharp_edges = preprocess_image(image)
        image = overlay_images(image, sharp_edges, OVERLAY_COUNT)

    # Run ML segmentation
    masks, outlines, _, _, _ = segment_cells(
        image, model_type=DEFAULT_MODEL_TYPE, 
        channels=DEFAULT_CHANNELS, use_gpu=USE_GPU )
    
    cell_areas, cell_centroids = extract_cell_properties(masks)
    
    # Plot segmented image 
    plot_segmentation(image, outlines, dpi=orginial_image_dpi, title="Cells Captured via Segmentation")
    
    ## Run analysis 
    # Create lines along the image
    line_positions = calculate_line_positions(image.shape, num_lines=NUM_LINES, axisSel=AXIS_SELECTION)
    
    # Detect cells along the lines in the image
    results = detect_cells_alongLines(masks, axis=AXIS_SELECTION, line_positions=line_positions)
    
    # Plot image with lines and detected cells along the lines
    plot_image_with_annotations(image, outlines, cell_centroids, results, line_positions, 
                            axisSel=AXIS_SELECTION, dpi=orginial_image_dpi, 
                            num_lines=NUM_LINES, img_file_name="image_withLinesAndCells.png", output_dir=output_dir)


    # Process and analyze cell distributions
    Dx_list, Dy_list, CellAreaPlot = [], [], []
    for line_index, cells_data in results.items():
        Dx_listD=[]
        Dy_listD=[]
        CellAreaPlotD=[]
        for cell_data in cells_data:
            CellAreaPlotD.append(float(cell_data[-1]))
            Dx_listD.append(cell_data[1])
            Dy_listD.append(cell_data[2])
            
        CellAreaPlot.append(CellAreaPlotD)
        Dx_list.append(Dx_listD)
        Dy_list.append(Dy_listD)
        
    # Plot distribution of Dx, Dy and cell areas for each line
    plot_distribution(CellAreaPlot, 'Cell Area Distribution', 'Cell Area (pixels)', nbins=NBINS, img_file_name="cAr_Distribution.png", output_dir=output_dir)
    plot_distribution(Dx_list, 'Dx Distribution', 'Dx (pixels)', nbins=NBINS, img_file_name="Dx_Distribution.png", output_dir=output_dir)
    plot_distribution(Dy_list, 'Dy Distribution', 'Dy (pixels)', nbins=NBINS, img_file_name="Dy_Distribution.png", output_dir=output_dir)


    # Save processed data
    with open(os.path.join(output_dir, "Dx_data.json"), "w") as f:
        json.dump(Dx_list, f)
    with open(os.path.join(output_dir, "Dy_data.json"), "w") as f:
        json.dump(Dy_list, f)
    with open(os.path.join(output_dir, "CellArea_data.json"), "w") as f:
        json.dump(CellAreaPlot, f)

    print(f"Processing complete for image >>> {image_name}")

if __name__ == "__main__":
    main()
