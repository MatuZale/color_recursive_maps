import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib import rcParams
from numba import njit

# Configure Matplotlib for high-quality output and LaTeX rendering
rcParams['text.usetex'] = True  # Set to False if LaTeX is unavailable
rcParams["text.latex.preamble"] = r"\usepackage{amsmath, amsfonts, systeme}"
rcParams['figure.dpi'] = 300  # High DPI for clear output

def create_colormap(hex_color1, hex_color2, hex_color3, reverse=False):
    """Create a custom linear segmented colormap from three hex colors."""
    colors = [to_rgb(hex_color1), to_rgb(hex_color2), to_rgb(hex_color3)]
    if reverse:
        colors = colors[::-1]
    return LinearSegmentedColormap.from_list("custom_colormap", colors)

@njit
def meshgrid(x, y):
    """Generate 2D meshgrid using Numba for performance."""
    xx = np.empty((y.size, x.size), dtype=x.dtype)
    yy = np.empty((y.size, x.size), dtype=y.dtype)
    for j in range(y.size):
        for k in range(x.size):
            xx[j, k] = x[k]
            yy[j, k] = y[j]
    return xx, yy

@njit
def calc_ikeda_orbit(n_points, a, b, n_iter, x_range=(-2, 2), y_range=(-2, 2)):
    """Compute orbits for the given map using Numba."""
    x = np.linspace(x_range[0], x_range[1], n_points)
    y = np.linspace(y_range[0], y_range[1], n_points)
    xx, yy = meshgrid(x, y)
    
    # Pre-allocate arrays for performance
    l_cx = np.zeros(n_iter * n_points**2)
    l_cy = np.zeros(n_iter * n_points**2)
    
    for i in range(n_iter):
        # Recursive equations
        xx_new = 2 * np.sin(xx ** 2 - yy ** 2 + a)
        yy_new = 2 * np.cos(2 * xx * yy + b)
        xx, yy = xx_new, yy_new
        l_cx[i * n_points**2:(i + 1) * n_points**2] = xx.ravel()
        l_cy[i * n_points**2:(i + 1) * n_points**2] = yy.ravel()
    
    return l_cx, l_cy

def plot_orbit(l_cx, l_cy, a, b, bins=3000, area=([-2, 2], [-2, 2]), cmap=None):
    """Generate and display the given map orbit plot with a histogram."""
    g, _, _ = np.histogram2d(l_cx, l_cy, bins=bins, range=area)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    # Display histogram with logarithmic scaling
    ax.imshow(np.log1p(g), vmin=0, vmax=5, cmap=cmap, origin='lower')
    ax.set_xticks([]), ax.set_yticks([])
    
    # LaTeX title with given map equations
    ax.set_title(
        r'$\begin{array}{lr} '
        r'x_{t+1} = 2\sin(x_t^2 - y_t^2 + %.2f ) \\ '
        r'y_{t+1} = 2\cos(2 x_n y_n + %.2f ) \\ '
        r'\end{array}$' % (a, b),
        y=0.06, fontsize=14, color="#FFFFFF"
    )
    
    plt.show()
    # Optional: Save the plot
    plt.savefig('map_plot.png', dpi=300, bbox_inches='tight')

def main():
    """Main function to set parameters and execute the given map plot."""
    # Parameters
    n_points = 700
    n_iter = 300
    a, b = 3.4415, 2.7282  # Parameter for chaotic behavior
    area = [[-2, 2], [-2, 2]]
    
    # Create custom colormap
    color_map = create_colormap("#FDA000","#0d3d3b", "#3BDCF1", reverse=True)
    
    # Compute given map orbits
    l_cx, l_cy = calc_ikeda_orbit(n_points, a, b, n_iter, x_range=area[0], y_range=area[1])
    
    # Plot the result
    plot_orbit(l_cx, l_cy, a, b, bins=3000, area=area, cmap=color_map)

if __name__ == "__main__":
    main()
