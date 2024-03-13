import cv2
import json
import os
import numpy as np

def extract_and_preprocess_roi(image_path, annotations, output_dir):
    # Read the image
    image = cv2.imread(image_path)
    annotation_id, resized_roi, category_id = None, None, None
    
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
            category_id = annotation['category_id']
            
            roi_filename = f"{annotation_id}.jpg"
            roi_path = os.path.join(output_dir, roi_filename)
            cv2.imwrite(roi_path, resized_roi)
            print(f"Saved preprocessed ROI: {roi_filename}")
            
    return annotation_id, resized_roi, category_id

# Set paths
input_dir =  "./projectData/ena24"  # path to input images directory
json_file = "./projectData/metadata.json" # path to json
grey_scale_arrays_npz_file = "./projectData/grey_scale_arrays.npz"
grey_scale_samples_and_category_ids_npz_file = "./projectData/grey_scale_samples_and_category_ids.npz"
output_dir = "./projectData/processed"  # path to output directory

# Load JSON file containing metadata
with open(json_file, 'r') as f:
    metadata = json.load(f)

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

grey_scale_arrays_and_ids = {}
grey_scale_samples = []
category_ids = []

# Iterate through images and extract preprocess ROIs, annotation ids and category ids
for image_info in metadata['images']:
    image_id = image_info['id']
    image_path = os.path.join(input_dir, image_info['file_name'])
    annotations = [ann for ann in metadata['annotations'] if ann['image_id'] == image_id]
    annotation_id, resized_roi, category_id = extract_and_preprocess_roi(image_path, annotations, output_dir)
    
    # store roi with annotation_id key
    grey_scale_arrays_and_ids[annotation_id] = resized_roi
    
    # flatten roi matrix into one row of grey scale features
    grey_scale_sample_features = resized_roi.flatten()
    
    # store each row of grey scale features
    grey_scale_samples.append(grey_scale_sample_features)
    
    # store corresponding category_id
    category_ids.append(category_id)


# save npz file of grey scale matrixes of each image stored as np.arrays
np.savez(grey_scale_arrays_npz_file, **grey_scale_arrays_and_ids)

# X: grey scale samples matrix (np.array) where each row is an image
# and that row's columns are the grey scale feature values
X = np.vstack(grey_scale_samples)

# y: np.array of the corresponding categories
y = np.array(category_ids).flatten() 
 
# np.arrays to save: sample matrix and category array to save
samples_and_categories_arrays = {'grey_scale_samples' : X,'category_ids' : y}

# save npz file of grey scale samples matrix and corresponding category vector, matrix X and vector y
# these are stored as np.arrays and can be easily accessed through the names 'grey_scale_samples' and
# 'category_ids' after loading, np.load(), the npz file
np.savez(grey_scale_samples_and_category_ids_npz_file, **samples_and_categories_arrays)