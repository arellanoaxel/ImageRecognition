import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA

grey_scale_samples_and_category_ids_npz_file = "./projectData/grey_scale_samples_and_category_ids.npz"

grey_scale_samples_and_category_ids_dic = np.load(grey_scale_samples_and_category_ids_npz_file)

X = grey_scale_samples_and_category_ids_dic['grey_scale_samples']
X = scale(X)
y = grey_scale_samples_and_category_ids_dic['category_ids']



bird = 0 # id 0
coyote = 20 # id 20
bear = 22 # id 22


wanted = [bear, bird]

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

# perform KMeans
n_catagories = 2
kmeans = KMeans(n_clusters=n_catagories, random_state=12, n_init=10)
predictions = kmeans.fit_predict(new_X)

color_themes = np.array(['black',
                         'red',
                         'orange',
                         'green',
                         'blue',
                         'm',
                         'lightpink',
                         'violet',
                         'cornflowerblue',
                         'lime',
                         'aquamarine',
                         'darkcyan',
                         'lightskyblue',
                         'palegreen',
                         'sienna',
                         'peachpuff',
                         'lightgrey',
                         'yellow',
                         'gold',
                         'c',
                         'thistle',
                         'palevioletred',
                         'darkgoldenrod'])

# use two features to plot and color their points with label/category color
feature_0 = new_X[:,0]
feature_1 = new_X[:,1]
fig, axs = plt.subplots(2)

# plot these so you can visuallly see how to reorder
# axs[0].scatter(x=feature_0, y=feature_1, c = color_themes[new_y])
# axs[1].scatter(x=feature_0, y=feature_1, c = color_themes[predictions])


# need to reorder cluster labels before we can calculate accuracy
predictions[predictions == 0] = 0
# predictions[predictions == 2] = 0
predictions[predictions == 1] = 22


plt.subplot(2,1,1)
# plot known labels/truth
plt.scatter(x=feature_0, y=feature_1, c = color_themes[new_y])
plt.title('Truth (Bears & birds)')

plt.subplot(2,1,2)
# plot predictions
plt.scatter(x=feature_0, y=feature_1, c = color_themes[predictions])
plt.title('Predictions')
print(f'Accuracy = {accuracy_score(new_y, predictions)}')
plt.show()

# use PCA to reduce to 2D and plot after fitting
reduced_data = PCA(n_components=2).fit_transform(new_X)
kmeans = KMeans(n_clusters=n_catagories, n_init=10)
kmeans.fit(reduced_data)

# Step size of the mesh
h = 0.02  

# decision boundary
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# labels for each point in mesh
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(
    Z,
    interpolation="nearest",
    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    cmap=plt.cm.Paired,
    aspect="auto",
    origin="lower",
)

plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
# Plot the centroids 
centroids = kmeans.cluster_centers_
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="x",
    s=169,
    linewidths=3,
    color="w",
    zorder=10,
)
plt.title(
    "K-means clustering on bears & birds dataset (PCA-reduced data)\n"
    "Centroids are marked with white X"
)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
    