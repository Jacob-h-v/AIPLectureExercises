import numpy as np
import matplotlib.pyplot as plt

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

# 1000 rows, 2 columns - each row a sample, each column a dimension
samples = np.random.multivariate_normal(mean, covariance_matrix, 1000)
samples2 = np.random.multivariate_normal(mean2, covariance_matrix2, 1000)
samplestest = np.random.multivariate_normal(mean_test, covariance_matrix_test, 1000)
# print(samples.shape)

# Compute the mean value over all samples in each dimension.
# Option 0 signifies that we want mean computed along first dimension (rows, i.e., samples)
actualmean = np.mean(samples,0)
print(actualmean)


def mahalanobis_distance(*point_clouds):
    # combine the point clouds
    combined_clouds = np.vstack(point_clouds)

    # Create a list of the given point clouds
    clouds_list = list(point_clouds)

    # Center the combined clouds around the origin
    centered_clouds = combined_clouds - np.mean(combined_clouds, axis=0)

    # Try calculating covariance matrix
    try:
        calc_cov_matrix = np.cov(centered_clouds, rowvar=False)
        print(f"Calculated Cov Matrix:\n {calc_cov_matrix}")
    except Exception as e:
        print(f"Error calculating covariance matrix: {e}")
        return None

    # Check if the covariance matrix is invertible (not singular)
    try:
        inv_cov_matrix = np.linalg.inv(calc_cov_matrix)
    except np.linalg.LinAlgError:
        print("Error inverting the covariance matrix: Covariance matrix is probably singular (the point clouds are probably collinear or nearly so.")
        return None

    # Initialize a list to store the distances
    mahalanobis_distances = []

    # Calculate the Mahalanobis distance for each pair of point clouds
    for i in range(len(point_clouds)):
        for j in range(i + 1, len(point_clouds)):
            # Calculate the difference between the two point clouds
            diff = point_clouds[i] - point_clouds[j]
            # Calculate the squared mahalanobis distance
            mahalanobis_sq = np.sum(np.dot(diff, inv_cov_matrix) * diff, axis=1)
            # Take square root to get actual distance
            mahalanobis_dist = np.sqrt(mahalanobis_sq)
            # Add the calculated distance to the list of distances
            mahalanobis_distances.append(mahalanobis_dist)

    # Return list of distances
    return mahalanobis_distances, clouds_list


# Function to calculate the Mahalanobis distance between a point and a mean
def mahalanobis_distance_point_mean(point, mean, cov_matrix):
    diff = point - mean
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    mahalanobis_sq = np.dot(diff, np.dot(inv_cov_matrix, diff))
    return np.sqrt(mahalanobis_sq)

# ... (your existing code for defining mahalanobis_distance function)

# Call the definition to calculate the mahalanobis distance between the point clouds
distances, samples_list = mahalanobis_distance(samples, samples2)

# ... (your existing code for checking the result of mahalanobis_distance)

# Calculate the mean of the given sample
mean_test_sample = np.mean(samplestest, axis=0)

# Calculate the Mahalanobis distance from the mean of the sample to the means of all point clouds
distances_to_means = [mahalanobis_distance_point_mean(mean_test_sample, np.mean(clouds, axis=0), np.cov(clouds, rowvar=False)) for clouds in samples_list]

# Determine the nearest mean
nearest_mean_index = np.argmin(distances_to_means)
nearest_mean = np.mean(samples_list[nearest_mean_index], axis=0)
nearest_color = ['blue', 'red', 'yellow', 'green'][nearest_mean_index] # (add more colors if needed for more clouds)

# Plot scatter plot
fig, ax = plt.subplots(figsize=(6,  6))

# Plot each cloud with a different color
for i, cloud in enumerate(samples_list):
    ax.scatter(cloud[:,  0], cloud[:,  1], marker=".", color=['blue', 'red', 'yellow', 'green'][i], alpha=0.5)

ax.scatter(samplestest[:,  0], samplestest[:,  1], marker=".", color=nearest_color, alpha=0.5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_box_aspect(1.0)
ax.set_xlim([0,  20])
ax.set_ylim([0,  20])

plt.show()

exit()