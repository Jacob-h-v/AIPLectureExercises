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


def mahalanobis_distance(point_cloud1, point_cloud2):
    # Center point clouds around the origin
    point_cloud1_centered = point_cloud1 - np.mean(point_cloud1, axis=0)
    point_cloud2_centered = point_cloud2 - np.mean(point_cloud2, axis=0)

    # combine the point clouds
    combined_clouds = np.vstack((point_cloud1_centered, point_cloud2_centered))

    # calculate covariance matrix
    calc_cov_matrix = np.cov(combined_clouds, rowvar=False)
    print("Calculated Cov Matrix:")

    # Check if the covariance matrix is invertible (not singular)
    try:
        inv_cov_matrix = np.linalg.inv(calc_cov_matrix)
    except np.linalg.LinAlgError:
        print("Error inverting the covariance matrix: Covariance matrix is probably singular (the point clouds are probably collinear or nearly so.")
        return None

    # Calculate the Mahalanobis distance
    diff = point_cloud1_centered - point_cloud2_centered
    mahalanobis_sq = np.sum(np.dot(diff, inv_cov_matrix) * diff, axis=1)
    mahalanobis_dist = np.sqrt(mahalanobis_sq)
    print(f"Calculated mahalanobis distance: {mahalanobis_dist}")
    return mahalanobis_dist


# Call the definition to calculate the mahalanobis distance between the point clouds
mahalanobis_distance(samples, samples2)

# Check whether the result is None. If it isn't, proceed as normal
if mahalanobis_distance is None:
    print("An error occoured while calculating the mahalanobis distance.")
else:
    print("Mahalanobis distance calculated successfully.")


# Plot scatter plot
fig, ax = plt.subplots(figsize=(6, 6))

ax.scatter(samples[:,0], samples[:,1], marker=".", color='blue')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_box_aspect(1.0)
ax.set_xlim([0, 20])
ax.set_ylim([0, 20])


# plt.show will halt thread - program cannot terminate until open windows have been closed
# plt.figure(figsize=(10,10))
plt.show()



exit()