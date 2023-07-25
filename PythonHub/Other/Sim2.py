import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse

def double_pendulum_simulation():
    # Simulation parameters
    length1 = 1.0
    length2 = 0.5
    mass1 = 1.0
    mass2 = 1.0
    g = 9.81
    time_step = 0.05
    duration = 10.0

    # Initial conditions
    theta1 = np.pi / 2
    theta2 = np.pi / 2
    omega1 = 0.0
    omega2 = 0.0

    # Arrays to store simulation data
    t = np.arange(0, duration, time_step)
    theta1_vals = []
    theta2_vals = []
    tip1_paths = []
    tip2_paths = []

    # Perform simulation using numerical integration (Runge-Kutta method)
    for _ in t:
        theta1_vals.append(theta1)
        theta2_vals.append(theta2)

        alpha1_num = -g * (2 * mass1 + mass2) * np.sin(theta1)
        alpha1_denom = -length1 * (2 * mass1 + mass2 - mass2 * np.cos(2 * theta1 - 2 * theta2))
        alpha1 = alpha1_num / alpha1_denom

        alpha2_num = 2 * np.sin(theta1 - theta2) * (mass1 + mass2) * (omega1 ** 2 * length1 + g * np.cos(theta1 - theta2) * mass1)
        alpha2_denom = length2 * (2 * mass1 + mass2 - mass2 * np.cos(2 * theta1 - 2 * theta2))
        alpha2 = alpha2_num / alpha2_denom

        omega1 += alpha1 * time_step
        omega2 += alpha2 * time_step
        theta1 += omega1 * time_step
        theta2 += omega2 * time_step

        # Calculate the positions of the tips of the pendulums
        x1 = length1 * np.sin(theta1)
        y1 = -length1 * np.cos(theta1)
        x2 = x1 + length2 * np.sin(theta2)
        y2 = y1 - length2 * np.cos(theta2)
        tip1_paths.append([x1, y1])
        tip2_paths.append([x2, y2])

    # Create the animation
    fig, ax = plt.subplots()
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])

    # Create rods and joints
    rod1 = plt.Line2D([], [], linewidth=2, color='b')
    rod2 = plt.Line2D([], [], linewidth=2, color='g')
    joint1 = Ellipse([0, 0], 0.05, 0.05, fc='b')
    joint2 = Ellipse([0, 0], 0.05, 0.05, fc='g')

    ax.add_line(rod1)
    ax.add_line(rod2)
    ax.add_patch(joint1)
    ax.add_patch(joint2)

    def update(frame):
        # Update positions of the rods and joints
        rod1.set_data([0, tip1_paths[frame][0]], [0, tip1_paths[frame][1]])
        rod2.set_data([tip1_paths[frame][0], tip2_paths[frame][0]], [tip1_paths[frame][1], tip2_paths[frame][1]])
        joint1.center = (0, 0)
        joint2.center = (tip1_paths[frame][0], tip1_paths[frame][1])

        return rod1, rod2, joint1, joint2

    anim = FuncAnimation(fig, update, frames=len(t), interval=50, blit=True)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Double Pendulum Simulation')
    plt.grid(True)
    plt.show()

# Run the simulation
double_pendulum_simulation()
