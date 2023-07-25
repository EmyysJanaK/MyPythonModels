import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def projectile_motion_simulation():
    # Simulation parameters
    angle = 45  # Launch angle in degrees
    initial_speed = 25  # Initial speed of the projectile
    time_step = 0.01  # Time step for simulation
    duration = 5  # Duration of the simulation in seconds

    # Calculate the motion of the projectile
    t = np.arange(0, duration, time_step)
    x = initial_speed * np.cos(np.deg2rad(angle)) * t
    y = initial_speed * np.sin(np.deg2rad(angle)) * t - 0.5 * 9.8 * t ** 2

    # Create the animation
    fig, ax = plt.subplots()
    ax.set_xlim([0, max(x) + 5])
    ax.set_ylim([0, max(y) + 5])
    line, = ax.plot([], [], 'bo')

    def update(frame):
        line.set_data(x[:frame], y[:frame])
        return line,

    anim = FuncAnimation(fig, update, frames=len(t), interval=50, blit=True)

    plt.xlabel('Horizontal Distance (m)')
    plt.ylabel('Vertical Distance (m)')
    plt.title('Projectile Motion Simulation')
    plt.grid(True)
    plt.show()

# Run the simulation
projectile_motion_simulation()
