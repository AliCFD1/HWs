import numpy as np
import matplotlib.pyplot as plt

# Define the domain
start = 0
end = 1
num_points = 500
step_size = (end - start) / num_points
points = np.linspace(start, end, num_points + 1)

# Define the heat source
def heat_source(x):
    return (5 / np.sqrt(2 * np.pi)) * np.exp(-12.5 * (x - 0.5)**2)

# Initialize the temperature distribution
temperature = np.zeros(num_points + 1)
temperature[0] = 1

# Convergence tolerance
tolerance = 1e-9

# Iteration count
iterations = 1

while True:
    new_temperature = np.copy(temperature)
    for i in range(1, num_points):
        new_temperature[i] = (-0.5) * (step_size**2 * heat_source(points[i - 1]) - temperature[i - 1] - temperature[i + 1])

    # Apply boundary conditions
    new_temperature[0] = 1
    new_temperature[num_points] = (1 / (1 + 0.1 * step_size)) * temperature[num_points - 1]
    
    if np.max(np.abs(new_temperature - temperature)) < tolerance:
        break
    
    temperature = new_temperature
    iterations += 1

# Plot the results
#plt.plot(points, temperature)
#plt.title('Temperature Distribution')
#plt.xlabel('Position (x)')
#plt.ylabel('Temperature (u)')
#plt.show()

# Display the temperature at each point
for i in range(num_points + 1):
    print(f'{points[i]:.3f} {temperature[i]:.5f}')

print(f'Number of iterations: {iterations}')
