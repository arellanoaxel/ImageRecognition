import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA


# load samples and labels
grey_scale_samples_and_category_ids_npz_file = "./projectData/grey_scale_samples_and_category_ids.npz"

grey_scale_samples_and_category_ids_dic = np.load(grey_scale_samples_and_category_ids_npz_file)

# get samples and labels from dic
X = grey_scale_samples_and_category_ids_dic['grey_scale_samples']
X = scale(X)
y = grey_scale_samples_and_category_ids_dic['category_ids']


# label/category ids
Bird = 0
Eastern_Gray_Squirrel = 1
Eastern_Chipmunk = 2
Woodchuck = 3
Wild_Turkey = 4
White_Tailed_Deer = 5
Virginia_Opossum = 6
Eastern_Cottontail = 7
Human = 8
Vehicle = 9
Striped_Skunk = 10
Red_Fox = 11
Eastern_Fox_Squirrel = 12
Northern_Raccoon = 13
Grey_Fox = 14
Horse = 15
Dog = 16
American_Crow = 17
Chicken = 18
Domestic_Cat = 19
Coyote = 20
Bobcat = 21
American_Black_Bear = 22

# list of wanted labels
wanted = [
    Bird,
    Eastern_Gray_Squirrel,
    Eastern_Chipmunk,
    Woodchuck,
    Wild_Turkey,
    White_Tailed_Deer,
    Virginia_Opossum,
    Eastern_Cottontail,
    Human,
    Vehicle,
    Striped_Skunk,
    Red_Fox,
    Eastern_Fox_Squirrel,
    Northern_Raccoon,
    Grey_Fox,
    Horse,
    Dog,
    American_Crow,
    Chicken,
    Domestic_Cat,
    Coyote,
    Bobcat,
    American_Black_Bear
    ]

def find_lowest_noise_dbscan_params(data):
    best_epsilon = None
    best_min_samples = None
    lowest_noise_ratio = float('inf')

    # Iterate over a range of epsilon and min_samples values
    for epsilon in np.arange(0.1, 1.0, 0.1):
        for min_samples in range(2, 10):
            dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
            clusters = dbscan.fit_predict(data)

            # Calculate the noise ratio
            noise_ratio = sum(clusters == -1) / len(clusters)

            # Update best parameters if a lower noise ratio is found
            if noise_ratio < lowest_noise_ratio:
                lowest_noise_ratio = noise_ratio
                best_epsilon = epsilon
                best_min_samples = min_samples

    return best_epsilon, best_min_samples



def find_best_dbscan_params(data, target_clusters):
    best_epsilon = None
    best_min_samples = None
    smallest_difference = float('inf')
    lowest_noise_ratio = float('inf')

    # Iterate over a range of epsilon and min_samples values
    for epsilon in np.arange(0.1, 1.0, 0.1):
        for min_samples in range(2, 10):
            dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
            clusters = dbscan.fit_predict(data)

            # Calculate the number of clusters and noise ratio
            num_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            noise_ratio = sum(clusters == -1) / len(clusters)
            difference = abs(num_clusters - target_clusters)

            # Update best parameters if better combination is found
            if difference < smallest_difference or (difference == smallest_difference and noise_ratio < lowest_noise_ratio):
                smallest_difference = difference
                lowest_noise_ratio = noise_ratio
                best_epsilon = epsilon
                best_min_samples = min_samples

            # Break early if exact match is found and noise is minimal
            if difference == 0 and noise_ratio < lowest_noise_ratio:
                return best_epsilon, best_min_samples

    return best_epsilon, best_min_samples

# make new sample matrix and category array (X,y) but only include
# the wanted categories
def make_sample_matrix_and_category_array(X,y,*wanted_categories):
    truth_index = []
    for category in wanted_categories:
        truth_index.append((y == category))
    
    wanted_indexes = np.where(np.logical_or.reduce(truth_index))[0]
    wanted_y = y[wanted_indexes]
    wanted_X = X[wanted_indexes]
    return wanted_X, wanted_y

new_X, new_y = make_sample_matrix_and_category_array(X,y,*wanted)

# perform DBScan
epsilon = 0.5  # This is the maximum distance between two samples for one to be considered in the neighborhood of the other. Adjust as necessary.
min_samples = 5  # Minimum number of samples in a neighborhood for a point to be considered a core point. Adjust as necessary.

dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
clusters = dbscan.fit_predict(new_X)
print(Counter(clusters))

# Handling noise points (labeled as -1)
noise_points = clusters == -1
clustered_points = ~noise_points


# used to color code labels (23 colors if needed)
color_themes = np.array([
    'black', 'red', 'orange', 'green', 'blue', 'm', 'lightpink',
    'violet', 'cornflowerblue', 'lime', 'aquamarine', 'darkcyan',
    'lightskyblue', 'palegreen', 'sienna', 'peachpuff', 'lightgrey',
    'yellow', 'gold', 'c', 'thistle', 'palevioletred', 'darkgoldenrod',
    'navy', 'maroon', 'olive', 'purple', 'teal', 'silver', 'fuchsia'
])
# use two features to plot and color their points with label/category color
feature_0 = new_X[:,0]
feature_1 = new_X[:,1]
fig, axs = plt.subplots(2)

# plot these initially so you can visually see how to reorder
# since K-means uses random labels that must be correcly matched afterwards
# axs[0].scatter(x=feature_0, y=feature_1, c = color_themes[new_y])
# axs[1].scatter(x=feature_0, y=feature_1, c = color_themes[predictions])


# # need to reorder cluster labels before we can calculate accuracy
# predictions[predictions == 0] = 0
# # predictions[predictions == 2] = 0
# predictions[predictions == 1] = 22


# use PCA to reduce to 2D and plot after fitting
reduced_data = PCA(n_components=2).fit_transform(new_X)





# Apply DBSCAN to PCA-reduced data
epsilon = 0.5  # adjust as necessary
min_samples = 5  # adjust as necessary
dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
clusters = dbscan.fit_predict(reduced_data)

# Handling noise points (labeled as -1)
noise_points = clusters == -1
clustered_points = ~noise_points

print(Counter(clusters))

# Plotting
# plt.figure()
# # plt.scatter(reduced_data[clustered_points, 0], reduced_data[clustered_points, 1], c=color_themes[clusters[clustered_points]], marker='o')
# # plt.scatter(reduced_data[noise_points, 0], reduced_data[noise_points, 1], c='black', marker='x', label='Noise')
# plt.title("DBSCAN clustering on PCA-reduced data")
# plt.xlabel("PCA Feature 0")
# plt.ylabel("PCA Feature 1")
# plt.legend()
# plt.show()

for i in range(2, 28):
    bepsilon, bmin_samples = find_best_dbscan_params(reduced_data, i)
    print(f"Target Clusters: {i}. \nEpsilon: {bepsilon}.\nMin-Samples: {bmin_samples}")
    dbscan = DBSCAN(eps=bepsilon, min_samples=bmin_samples)
    clusters = dbscan.fit_predict(reduced_data)

    # Handling noise points (labeled as -1)
    noise_points = clusters == -1
    clustered_points = ~noise_points

    print(Counter(clusters))

    # Plotting
    plt.figure()
    plt.scatter(reduced_data[noise_points, 0], reduced_data[noise_points, 1], c='black', marker='x', label='Noise')
    plt.scatter(reduced_data[clustered_points, 0], reduced_data[clustered_points, 1], c=color_themes[clusters[clustered_points]], marker='o')
    #
    plt.title("DBSCAN clustering on PCA-reduced data")
    plt.xlabel("PCA Feature 0")
    plt.ylabel("PCA Feature 1")
    plt.legend()
    plt.show()


bepsilon, bmin_samples = find_lowest_noise_dbscan_params(reduced_data)


dbscan = DBSCAN(eps=bepsilon, min_samples=bmin_samples)
clusters = dbscan.fit_predict(reduced_data)

# Handling noise points (labeled as -1)
noise_points = clusters == -1
clustered_points = ~noise_points

print(Counter(clusters))

# Plotting
plt.figure()
plt.scatter(reduced_data[noise_points, 0], reduced_data[noise_points, 1], c='black', marker='x', label='Noise')
plt.scatter(reduced_data[clustered_points, 0], reduced_data[clustered_points, 1], c=color_themes[clusters[clustered_points]], marker='o')
#
plt.title("DBSCAN clustering on PCA-reduced data")
plt.xlabel("PCA Feature 0")
plt.ylabel("PCA Feature 1")
plt.legend()
plt.show()