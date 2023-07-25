import numpy as np
import matplotlib.pyplot as plt

def logistic_growth_simulation(r, K, P0, timesteps):
    population = [P0]
    dt = 1  # Time step

    for t in range(1, timesteps):
        dP_dt = r * population[-1] * (1 - population[-1] / K)
        P_next = population[-1] + dP_dt * dt
        population.append(P_next)

    return population

# Parameters for the simulation
growth_rate = 0.1
carrying_capacity = 1000
initial_population = 10
simulation_time = 100

# Run the simulation
population_over_time = logistic_growth_simulation(growth_rate, carrying_capacity, initial_population, simulation_time)

# Plot the results
time_steps = np.arange(0, simulation_time, 1)
plt.plot(time_steps, population_over_time)
plt.xlabel('Time')
plt.ylabel('Population Size')
plt.title('Logistic Growth Simulation')
plt.grid(True)
plt.show()
