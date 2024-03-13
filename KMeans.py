import numpy as np

grey_scale_arrays_npz_file = "./projectData/grey_scale_arrays.npz"

# load npz into dic
grey_scale_arrays_npz_dic = np.load(grey_scale_arrays_npz_file)

# get list of array names (ids)
grey_scale_array_ids = grey_scale_arrays_npz_dic.files

i = 1
for id in grey_scale_array_ids:
    # get an array based on name/id
    grey_scale_array = grey_scale_arrays_npz_dic[id]
    
    
    # just showing that these are correct for first 4 images
    print(id) # printing id
    print(grey_scale_array.shape) # showing shape is correct (64,64)
    print(grey_scale_array, end = '\n\n') # print the gray scale array
    if i == 4:
        break
    i += 1
    