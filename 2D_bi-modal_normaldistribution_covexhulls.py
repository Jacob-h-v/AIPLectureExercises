import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

# Check whether plots are interactive or not ... interactive mode is most useful from the command line
if plt.isinteractive() == True:
    print("Interactive mode is TRUE - exiting.")
    exit()
else:
    print("interactive mode is FALSE - remember to close graph window.")


# Generate 1000 samples from a 2D normal distribution
mean = [5.0, 8.0]
covariance_matrix = [[1, 0],
                     [0, 1]]

mean2 = [10.0, 14.0]
covariance_matrix2 = [[2, 1],
                      [1, 2]]

mean_test = [8.0, 9.0]
covariance_matrix_test = [[1, 0],
                          [0, 1]]

#1000 rows, 2 columns - each row a sample, each column a dimension
samples = np.random.multivariate_normal(mean, covariance_matrix, 1000)
samples2 = np.random.multivariate_normal(mean2, covariance_matrix2, 1000)
samplestest = np.random.multivariate_normal(mean_test, covariance_matrix_test, 1000)
#print(samples.shape)

#Compute the mean value over all samples in each dimension.
#Option 0 signifies that we want mean computed along first dimension (rows, i.e., samples)
actualmean = np.mean(samples,0)
print(actualmean)

# Compute distance between two points
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1- point2) ** 2))


# Function to compute convex hull for a set of points
def compute_convex_hull(points):
    hull = ConvexHull(points)
    return points[hull.vertices]


# Compute convex hulls for each set of samples
convex_hull_samples = compute_convex_hull(samples)
convex_hull_samples2 = compute_convex_hull(samples2)
convex_hull_samplestest = compute_convex_hull(samplestest)

# Calculate distances between convex hulls
distances_to_samples = np.min(cdist(convex_hull_samplestest, convex_hull_samples), axis=1)
distances_to_samples2 = np.min(cdist(convex_hull_samplestest, convex_hull_samples2), axis=1)

# Plot the existing scatterplots
plt.figure(figsize=(8, 6))
plt.scatter(samples[:, 0], samples[:, 1], color='blue', label='Samples 1')
plt.scatter(samples2[:, 0], samples2[:, 1], color='red', label='Samples 2')

# Plot the third scatterplot with color based on distances
plt.scatter(samplestest[:, 0], samplestest[:, 1], c=np.minimum(distances_to_samples, distances_to_samples2), cmap='viridis', label='Samplestest')

# Plot scatter plot
fig, ax = plt.subplots(figsize=(6, 6))

ax.scatter(samples[:,0], samples[:,1], marker=".", color='blue')
ax.scatter(samples2[:,0], samples2[:,1], marker=".", color='red')
plt.scatter(samplestest[:, 0], samplestest[:, 1], c=np.minimum(distances_to_samples, distances_to_samples2), cmap='viridis', label='Samplestest')
#ax.scatter(samplestest[:,0], samplestest[:,1], marker=".", color='yellow')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_box_aspect(1.0)
ax.set_xlim([0, 20])
ax.set_ylim([0, 20])


#plt.show will halt thread - program cannot terminate until open windows have been closed
#plt.figure(figsize=(10,10))
plt.show()



exit()