import numpy as np

# making random datasets
X = np.array([[2, 3], [3, 4], [1, 2], [5, 6], [6, 5], [7, 8]]) 
y = np.array([-1, -1, -1, 1, 1, 1])  

# Step 1: Initialize 
w = np.zeros(X.shape[1])  
b = 0
learning_rate = 0.01
lambda_param = 0.01  
epochs = 1000

# Step 2: Gradient Descent training
for epoch in range(epochs):
    for i in range(len(X)):
        # Compute the functional margin: y(w·x + b)
        if y[i] * (np.dot(X[i], w) + b) >= 1:
            w -= learning_rate * (2 * lambda_param * w)
        else:
            w -= learning_rate * (2 * lambda_param * w - np.dot(y[i], X[i]))
            b -= learning_rate * y[i]



# Step 3: Test the trained SVM
test_samples = np.array([[4, 5], [2, 2], [6, 6]])

# Step 4: predict function definition
predictions = np.sign(np.dot(test_samples, w) + b)

print("Trained Weights:", w)
print("Trained Bias:", b)
print("Predictions for test samples:", predictions)