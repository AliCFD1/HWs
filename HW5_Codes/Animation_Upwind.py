import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define constants
a = 0.1
L = 1.0
N = 201
dx = L / (N - 1)
dt = 0.5 * dx / a
x = np.linspace(0, L, N)
T = 2.5

# Define initial condition
def u0(x):
    return np.sin(2 * np.pi * x)

# Implement upwind discretization scheme
def upwind_update(u, a, dt, dx):
    u_new = np.zeros_like(u)
    for i in range(1, N - 1):
        if a > 0:
            u_new[i] = u[i] - a * dt / dx * (u[i] - u[i - 1])
        else:
            u_new[i] = u[i] - a * dt / dx * (u[i + 1] - u[i])

    # Apply periodic boundary conditions
    u_new[0] = u_new[-2]
    u_new[-1] = u_new[1]

    return u_new

# Initialize the solution array
u = u0(x)

# Create initial plot
fig, ax = plt.subplots()
ax.set_xlim(0, L)
ax.set_ylim(-1.5, 1.5)
line, = ax.plot(x, u0(x), label="Initial wave shape")
ax.set_xlabel("x")
ax.set_ylabel("u(x, t)")
ax.legend()

# Function to update the plot in each animation frame
def update(frame, u, line, a, dt, dx):
    t = frame * dt
    u[:] = upwind_update(u, a, dt, dx)
    line.set_ydata(u)
    ax.set_title(f'Time: {t:.2f}')

# Animation function
animation = FuncAnimation(fig, update, frames=int(T/dt), fargs=(u, line, a, dt, dx), interval=5, blit=False)

# Show the animation
plt.show()
