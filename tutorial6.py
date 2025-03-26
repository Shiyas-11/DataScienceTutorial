import numpy as np

# Sample 2D matrix dataset
X = np.array([
    [1, 2, 3, 4], 
    [5, 6, 7, 8], 
    [9, 10, 11, 12], 
    [13, 14, 15, 16]
])
# Step 1: Compute the Distance Matrix
def euc_dist(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

n = len(X)
dist_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(i + 1, n):
        dist_matrix[i, j] = dist_matrix[j, i] = euc_dist(X[i], X[j])

# Step 2: Initialize clusters 
clusters = {i: [i] for i in range(n)}

# Step 3: while loop to do the task
while len(clusters) > 1:
    # Find the two closest clusters (based on complete linkage)
    min_dist = float('inf')
    to_merge = (-1, -1)
    
    keys = list(clusters.keys())  
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            c1, c2 = keys[i], keys[j]
            # Compute complete linkage distance (maximum since complet linkage)
            max_dist = max(dist_matrix[p1, p2] for p1 in clusters[c1] for p2 in clusters[c2])
            
            if max_dist < min_dist:
                min_dist = max_dist
                to_merge = (c1, c2)

    # Merge the two clusters
    c1, c2 = to_merge
    clusters[c1].extend(clusters[c2])  # Merge points
    del clusters[c2]

    # new dist matrix
    for k in clusters.keys():
        if k != c1:
            new_dist = max(dist_matrix[p1, p2] for p1 in clusters[c1] for p2 in clusters[k])
            dist_matrix[c1, k] = dist_matrix[k, c1] = new_dist

    print(f"Merged clusters {c1} and {c2}, Remaining Clusters: {clusters}")

# Final cluster output
print("Final Cluster:", clusters)