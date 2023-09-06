import numpy as np
import matplotlib.pyplot as plt

ref = np.array((1, 1))
epsilon = 0.5
num_points = 1000
# Define a function to check Pareto dominance
def is_pareto_dominant(x, y):
    return x >= ref[0] and y >= ref[1]

# Define the grid dimensions
x_range = np.linspace(-2, 5, num_points)  # Values from 0 to 2 with 100 points
y_range = np.linspace(-2, 5, num_points)  # Values from 0 to 2 with 100 points

# Create a meshgrid from x and y ranges
xx, yy = np.meshgrid(x_range, y_range)

# Calculate the norm for each point in the grid relative to (1, 1)
distance_to_center = np.sqrt((xx - ref[0])**2 + (yy - ref[1])**2)
condition_mask = np.zeros((num_points, num_points), dtype=bool)

for x_idx, x in enumerate(x_range):
    for y_idx, y in enumerate(y_range):
        val = np.sqrt(x**2 + y**2) * epsilon
        if is_pareto_dominant(x, y):
            condition_mask[x_idx, y_idx] = 1
        elif x >= ref[0]:
            condition_mask[x_idx, y_idx] = np.sqrt((y - ref[1])**2) < val
        elif y >= ref[1]:
            condition_mask[x_idx, y_idx] = np.sqrt((x - ref[0])**2) < val
        else:
            condition_mask[x_idx, y_idx] = distance_to_center[x_idx, y_idx] < val

# Create a mask for points that Pareto dominate (1, 1)
pareto_mask = np.vectorize(is_pareto_dominant)(xx, yy)

# Combine the masks to determine the color of each point
color_mask = condition_mask | pareto_mask

# Create a plot
plt.figure(figsize=(8, 8))
plt.imshow(np.zeros_like(color_mask), extent=(0, 2, 0, 2), origin='lower', cmap='gray')  # Initialize with a black background
plt.colorbar(label='Satisfies Condition / Pareto Dominates (1, 1)')

# Plot points that do not satisfy the condition in red
plt.scatter(xx[~color_mask], yy[~color_mask], color='red', label='Does Not Satisfy Condition')

# Plot points that satisfy the condition in green
plt.scatter(xx[condition_mask], yy[condition_mask], color='green', label='Satisfies Condition')

# Plot points that Pareto dominate (1, 1) in green
#plt.scatter(xx[pareto_mask], yy[pareto_mask], color='green', label='Pareto Dominates (1, 1)')

plt.scatter(ref[0], ref[1], color='black', label='Referent')

# Add labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Points in the nonnegative region')

# Show the plot
#plt.legend()
plt.show()
