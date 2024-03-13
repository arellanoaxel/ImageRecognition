import cv2
import json
import os
import numpy as np

def extract_and_preprocess_roi(image_path, annotations, output_dir):
    # Read the image
    image = cv2.imread(image_path)
    annotation_id, resized_roi = None, None
    
    # Iterate through annotations
    for annotation in annotations:
        if annotation['image_id'] == image_id:
            # Get bounding box coordinates
            bbox = annotation['bbox']
            x, y, w, h = map(int, bbox)
            
            # Extract ROI
            roi = image[y:y+h, x:x+w]
            
            # Convert ROI to grayscale
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Resize ROI to 64x64 pixels
            resized_roi = cv2.resize(gray_roi, (64, 64))
            
            # Save the preprocessed ROI
            annotation_id = annotation['id']
            roi_filename = f"{annotation_id}.jpg"
            roi_path = os.path.join(output_dir, roi_filename)
            cv2.imwrite(roi_path, resized_roi)
            print(f"Saved preprocessed ROI: {roi_filename}")
            
    return annotation_id, resized_roi

# Set paths
input_dir =  "./projectData/ena24"  # path to input images directory
json_file = "./projectData/metadata.json" # path to json
grey_scale_arrays_npz_file = "./projectData/grey_scale_arrays.npz"
output_dir = "./projectData/processed"  # path to output directory

# Load JSON file containing metadata
with open(json_file, 'r') as f:
    metadata = json.load(f)

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

grey_scale_arrays_and_ids = {}

# Iterate through images and extract & preprocess ROIs
for image_info in metadata['images']:
    image_id = image_info['id']
    image_path = os.path.join(input_dir, image_info['file_name'])
    annotations = [ann for ann in metadata['annotations'] if ann['image_id'] == image_id]
    annotation_id, resized_roi = extract_and_preprocess_roi(image_path, annotations, output_dir)
    grey_scale_arrays_and_ids[annotation_id] = resized_roi

np.savez(grey_scale_arrays_npz_file, **grey_scale_arrays_and_ids)