import numpy as np
import matplotlib.pyplot as plt

def generate_pascals_triangle(rows):
    triangle = np.zeros((rows, rows), dtype=int)
    for i in range(rows):
        for j in range(i + 1):
            if j == 0 or j == i:
                triangle[i][j] = 1
            else:
                triangle[i][j] = triangle[i - 1][j - 1] + triangle[i - 1][j]
    return triangle

def plot_pascals_triangle(triangle):
    rows, cols = triangle.shape
    fig, ax = plt.subplots()

    for i in range(rows):
        for j in range(i + 1):
            x = (j - i / 2) * 20
            y = -i * 30
            ax.text(x, y, str(triangle[i][j]), ha='center')

    ax.axis('off')
    plt.title("Pascal's Triangle", fontsize=16)
    plt.show()

# Number of rows in Pascal's Triangle
rows = 10

# Generate and plot Pascal's Triangle
triangle = generate_pascals_triangle(rows)
plot_pascals_triangle(triangle)
