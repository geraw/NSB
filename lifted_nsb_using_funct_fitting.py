import math
from sympy import *
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

n = 3  # Number of robots
m = 5  # Number of lookahead time steps
DELTA = 1  # Sampling interval
objective = "Formation" # Baricenter or Formation or Obstacles

"""# Process Dynamics

##
We define the update function $up\colon \mathbb{R}^{n \times 2} \times  \mathbb{R}^{m \times n \times 2} \to \mathbb{R}^{n \times 2}$ by: $$up(r,v)=r+\Delta \sum_{i=0}^m v_i$$
"""

# Define the position update function
def up(robots, speed_vectors):
    return robots + DELTA * sum(speed_vectors)

# Test the function
robots = np.array([[1, 1], [2, 2], [3, 3]])
speed_vectors = np.array(
    [[[1, 1], [2, 2], [3, 3]], [[1, -2], [2, -4], [3, -6]], [[-2, 1], [-4, 2], [-6, 3]]]
)

assert np.allclose(up(robots, speed_vectors), robots)

"""##
Define the barycenter objective function $f\colon \mathbb{R}^{n \times 2} \to \mathbb{R}^{l}$, where $l=2$ as: $$f(r) := \frac{1}{n}\sum_{i=1}^{n} r_i$$
"""

################# Define the barycenter objective function
def barycenter(robots):
    return sum(robots) / len(robots)


# Test that the barycenter is correct
points = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
b = barycenter(points)
assert np.allclose(b, np.array([4, 5]))


################# Define the formation objective functions
def formation(robots):
    if len(robots) < 2:
        return robots  # Return as it is if only one point or empty list

    first_point = robots[0]  # Get the first point
    mapped_points = []

    for point in robots[1:]:
        difference = [point[0] - first_point[0], point[1] - first_point[1]]
        mapped_points.append(difference)

    return np.array(mapped_points).flatten()

# Test that the formation function is correct
points = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
form = formation(points)
assert np.allclose(form, np.array([2, 2, 4, 4, 6, 6]))


################# Define the obstacles objective functions

# Obstacle positions are given as a list of tuples
obstacles_pos = [(1, 1), (2, 3), (0, 0), (0, 2)]

# Threshold distance to obstacles
threshold = 0.2

# This function returns the distance to the closest obstacle for each robot
def obstacles(robots):
    distances = np.zeros(len(robots))

    for i, robot in enumerate(robots):
        min_distance = float("inf")

        for obstacle in obstacles_pos:
            distance = np.linalg.norm(robot - obstacle)
            if distance < threshold and distance < min_distance:
                min_distance = distance

        distances[i] = min_distance

    # Replace all float("inf") in distances with zero
    return np.where(distances == float("inf"), 0, distances).sum()

# Test that the compute_accumulated_distances function is correct
robots = np.array([[0.01, 0.02], [0.03, 0.04], [0.05, 0.06], [1.97, 2.98]])
assert np.allclose(obstacles(robots), 0.186518)


objectives = {"Baricenter": barycenter, "Formation": formation, "Obstacles": obstacles}
ls = {"Baricenter": 2, "Formation": 2*(n-1), "Obstacles": 1}

# This is our only objective function for the time being
f = objectives[objective]
l = ls[objective]

"""#####
The function that we are going to fit is $\delta\colon \mathbb{R}^{n \times 2} \times  \mathbb{R}^{m \times n \times 2} \to \mathbb{R}^{l}$ defined by
$$\delta(r,v) := f(up(r,v)) - f(r)$$
"""

def delta(r, v):
    return tuple(f(up(r, v)) - f(r))

# Test that the delta function is correct
r = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
v = np.array([[[1, 1], [2, 2], [3, 3], [4, 4]]])

if objective == "Baricenter":
    expected_result = np.array([[2.5, 2.5]])
elif objective == "Formation":
    expected_result = np.array([[1, 1, 2, 2, 3, 3]])

result = delta(r, v)
assert np.allclose(result, expected_result), f"Unexpected result: {result}, expected: {expected_result}"

"""# Model Fitting using PyTorch

## Data Generation
"""

num_samples = 5000

# Generate p and v
p = torch.randn(num_samples, 2*n)
v = torch.randn(num_samples, 2*n*m)

# Example of input to delta
pp = p.reshape(-1, n, 2)
vv = v.reshape(-1, m, n, 2)

# Calculate delta for each sample and convert to PyTorch tensors
fpv = torch.stack([torch.tensor(delta(pp[i], vv[i]), dtype=torch.float32) for i in range(num_samples)])

assert fpv[1].shape == (l,), "Unexpected shape of fpv"

batch_size = 100  # Batch size for training

assert batch_size <= num_samples, "Batch size must be less than or equal to the number of samples"

# Define the architecture of the neural network
class FunctionApproximator(nn.Module):
    def __init__(self):
        super(FunctionApproximator, self).__init__()
        self.create_mat = nn.Sequential(
            nn.Linear(2*n, 2*n),
            nn.ReLU(),
            nn.Linear(2*n, l * m * n * 2),
            nn.ReLU(),
        )
        for module in self.create_mat.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, p, v):
        output = self.create_mat(p)

        # Reshape the output into the desired matrix shape
        self.matrix = output.view(-1, l, m * n * 2)

        # Multiply the matrix by v
        output = torch.bmm(self.matrix, v.unsqueeze(-1)).squeeze(-1)

        return output

model = FunctionApproximator()

# Define the loss function
criterion = nn.MSELoss()

# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Convert data to batches
num_batches = num_samples // batch_size
print(f"{num_samples} samples")
print(f"{num_batches} batches")
p_batches = torch.split(p[:num_batches * batch_size], batch_size)
v_batches = torch.split(v[:num_batches * batch_size], batch_size)
fpv_batches = torch.split(fpv[:num_batches * batch_size], batch_size)

# Training loop
num_epochs = 20000
for epoch in range(num_epochs):
    total_loss = 0.0
    for batch in range(num_batches):
        # Forward pass
        outputs = model(p_batches[batch], v_batches[batch])

        # Compute the loss
        loss = criterion(outputs, fpv_batches[batch])
        total_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print progress
    if (epoch + 1) % 1000 == 0:
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}")

for _ in range(100):
    # Evaluation
    with torch.no_grad():
        # Generate test data
        test_p = torch.randn(1, 2*n)
        test_v = torch.randn(1, m*n*2)

        # Apply the trained model to compute J(p)v
        Jpv = model(test_p, test_v)

        # Print the result
        print("Approximated J(p)v:")
        print(tuple(Jpv[0].numpy()))

        print("\nReal delta(p,v):")
        pp = test_p.reshape(-1, n, 2).numpy()
        vv = test_v.reshape(-1, m, n, 2).numpy()
        print(delta(pp[0], vv[0]))



        # Extract the internal matrix produced for specific p
        print("\nThe matrix J(p):")
        mat = model.matrix

        # Print the tensor in a matrix format
        rows = mat.squeeze().tolist()
        for row in rows:
            print('\t'.join([f'{x:.4f}' for x in row]))

