# Linear Discriminant Analysis (LDA) using a stepwise approach for a 2x2 matrix
import numpy as np

# Define sample data matrices for two classes
class1 = np.array([[2, 3], [5, 7]])
class2 = np.array([[8, 6], [4, 10]])

# Calculate the mean vectors for each class
mean_class1 = np.mean(class1, axis=0).reshape(-1, 1)
mean_class2 = np.mean(class2, axis=0).reshape(-1, 1)

# Compute the within-class scatter matrix
scatter_within = np.dot((class1 - mean_class1), (class1 - mean_class1).T) + np.dot((class2 - mean_class2), (class2 - mean_class2).T)

# Compute the between-class scatter matrix
mean_difference = mean_class1 - mean_class2
scatter_between = np.dot(mean_difference, mean_difference.T)

# Calculate the discriminant vector
scatter_within_inv = np.linalg.inv(scatter_within)
discriminant_vector = np.dot(scatter_within_inv, mean_difference)

# Project data points onto the new axis
projection_class1 = np.dot(class1, discriminant_vector)
projection_class2 = np.dot(class2, discriminant_vector)

# Display the results
print(f"Class 1 Mean:\n{mean_class1}")
print(f"Class 2 Mean:\n{mean_class2}")
print(f"Within-Class Scatter Matrix:\n{scatter_within}")
print(f"Between-Class Scatter Matrix:\n{scatter_between}")
print(f"Discriminant Vector:\n{discriminant_vector}")
print(f"Class 1 Projections:\n{projection_class1}")
print(f"Class 2 Projections:\n{projection_class2}")
