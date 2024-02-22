import numpy as np


def read_csv(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
    # Skip header line
    data = [list(map(float, line.strip().split(","))) for line in lines[1:]]
    # Reshape each face into a two-dimensional array
    faces = [np.array(face).reshape(-1, 2) for face in data]
    return faces


def calculate_mean_and_cov(face):
    # Calculate the mean of the face
    mean = np.mean(face, axis=0)

    # Calculate the face's covariance matrix
    covariance_matrix = np.cov(face, rowvar=False)

    return mean, covariance_matrix


def mahalanobis_distance(face1, face2):
    mean1, cov1 = calculate_mean_and_cov(face1)
    mean2, cov2 = calculate_mean_and_cov(face2)

    # Check if the covariance matrix is two-dimensional and invertible
    if cov1.ndim != 2 or np.linalg.det(cov1) == 0:
        print("Covariance matrix for face1 is not invertible.")
        return float('inf'), float('inf')
    if cov2.ndim != 2 or np.linalg.det(cov2) == 0:
        print("Covariance matrix for face2 is not invertible.")
        return float('inf'), float('inf')

    # Calculate the inverse covariance matrices for face 1 and 2
    inv_cov1 = np.linalg.inv(cov1)
    inv_cov2 = np.linalg.inv(cov2)

    # Calculate the mahalanobis distances
    diff1 = mean1 - mean2
    diff2 = mean2 - mean1
    mahalanobis_distance1 = np.sqrt(np.dot(diff1, np.dot(inv_cov1, diff1)))
    mahalanobis_distance2 = np.sqrt(np.dot(diff2, np.dot(inv_cov2, diff2)))

    return mahalanobis_distance1, mahalanobis_distance2


def compare_new_face(new_face_file_path, existing_data1, existing_data2):
    # Read and pre-process data from new face file
    new_data = read_csv(new_face_file_path)

    # Calculate the mahalanobis distances to each existing file
    similarity_to_negative_face = mahalanobis_distance(new_data[0], existing_data1[0])
    similarity_to_positive_face = mahalanobis_distance(new_data[0], existing_data2[0])

    # Determine which file is more similar
    if similarity_to_negative_face[0] < similarity_to_positive_face[0]:
        print(f"The new face is more similar to the negative face")
    elif similarity_to_negative_face[0] > similarity_to_positive_face[0]:
        print(f"The new face is more similar to the positive face")
    else:
        print(f"The new face is equally similar to the positive and negative faces")


# Call the definitions
negative_face_path = ""
positive_face_path = ""
new_face_path = ""
negative_face_data = read_csv(negative_face_path)
positive_face_data = read_csv(positive_face_path)

compare_new_face(new_face_path, negative_face_data, positive_face_data)
