import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib import animation
from numba import njit
import time

@njit
def clifford_fast(n_points, a, b, c, d, n_iter):
    """Fast Clifford attractor computation."""
    # Use random initial points for faster computation
    x = np.random.uniform(-0.5, 0.5, n_points)
    y = np.random.uniform(-0.5, 0.5, n_points)
    
    # Store all trajectory points
    all_x = np.zeros(n_points * n_iter)
    all_y = np.zeros(n_points * n_iter)
    
    for i in range(n_iter):
        # Clifford map equations
        x_new = np.sin(a * y) + c * np.cos(a * x)
        y_new = np.sin(b * x) + d * np.cos(b * y)
        x, y = x_new, y_new
        
        # Store points
        start_idx = i * n_points
        end_idx = (i + 1) * n_points
        all_x[start_idx:end_idx] = x
        all_y[start_idx:end_idx] = y
    
    return all_x, all_y

def create_fast_clifford_video(duration_seconds=60, fps=24):
    """Create a fast, high-quality Clifford attractor video."""
    
    total_frames = duration_seconds * fps
    print(f"Creating {duration_seconds}s video at {fps} FPS = {total_frames} frames")
    
    # Optimized parameters for speed and quality
    n_points = 5000     # Number of trajectory points
    n_iter = 100        # Iterations per point
    bins = 800          # Histogram resolution
    
    # Parameter evolution - animate parameter 'a' for dramatic effect
    t = np.linspace(0, 6*np.pi, total_frames)  # 3 full cycles
    a_values = -1.4 + 0.8 * np.sin(t)  # Varies from -2.2 to -0.6
    
    # Fixed parameters
    b, c, d = 1.6, 1.0, 0.7
    
    # Beautiful colormap
    colors = ['#000033', '#0066CC', '#00FFFF', '#FFFF00', '#FF6600', '#FF0033']
    cmap = LinearSegmentedColormap.from_list("clifford", [to_rgb(c) for c in colors])
    
    # Set up figure
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('black')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    def animate(frame):
        ax.clear()
        ax.set_facecolor('black')
        
        # Current parameter value
        a = a_values[frame]
        
        # Progress indicator
        if frame % fps == 0:  # Every second
            print(f"Frame {frame+1}/{total_frames} ({frame/fps:.1f}s) - a={a:.3f}")
        
        # Compute attractor
        x_points, y_points = clifford_fast(n_points, a, b, c, d, n_iter)
        
        # Create density plot
        hist, xedges, yedges = np.histogram2d(x_points, y_points, bins=bins, 
                                            range=[[-3, 3], [-3, 3]])
        
        # Plot with logarithmic scaling
        im = ax.imshow(np.log1p(hist.T), origin='lower', 
                      extent=[-3, 3, -3, 3], cmap=cmap, 
                      vmin=0, vmax=np.log1p(hist.max()))
        
        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add title and info
        ax.text(0.5, 0.95, 'Clifford Attractor Evolution', 
               transform=ax.transAxes, ha='center', va='top',
               fontsize=16, color='white', weight='bold')
        
        ax.text(0.02, 0.02, f'a = {a:.3f}\nb = {b:.1f}\nc = {c:.1f}\nd = {d:.1f}', 
               transform=ax.transAxes, va='bottom',
               fontsize=12, color='white',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        return [im]
    
    print("Starting video generation...")
    start_time = time.time()
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=total_frames, 
                                 interval=1000/fps, blit=True)
    
    # Show a preview
    plt.show()
    
    # Save video
    filename = f'clifford_evolution_{duration_seconds}s_{fps}fps.mp4'
    
    try:
        print(f"Saving video: {filename}")
        print("This may take 5-15 minutes...")
        
        # High quality MP4
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, bitrate=6000, extra_args=['-vcodec', 'libx264'])
        anim.save(filename, writer=writer, dpi=120)
        
        elapsed = time.time() - start_time
        print(f"\n✅ SUCCESS!")
        print(f"📁 Video saved: {filename}")
        print(f"⏱️  Total time: {elapsed/60:.1f} minutes")
        print(f"🎬 Duration: {duration_seconds}s at {fps} FPS")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        
        # Fallback to GIF
        try:
            gif_name = f'clifford_evolution_{duration_seconds}s.gif'
            print(f"Saving as GIF: {gif_name}")
            anim.save(gif_name, writer='pillow', fps=fps//2)
            print(f"✅ GIF saved: {gif_name}")
        except Exception as e2:
            print(f"❌ GIF failed too: {e2}")
    
    return anim

# Quick test function
def test_single_frame():
    """Test with a single frame to verify everything works."""
    print("Testing single frame...")
    
    # Compute one frame
    x_points, y_points = clifford_fast(5000, -1.4, 1.6, 1.0, 0.7, 100)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    hist, _, _ = np.histogram2d(x_points, y_points, bins=800, range=[[-3, 3], [-3, 3]])
    
    colors = ['#000033', '#0066CC', '#00FFFF', '#FFFF00', '#FF6600', '#FF0033']
    cmap = LinearSegmentedColormap.from_list("test", [to_rgb(c) for c in colors])
    
    ax.imshow(np.log1p(hist.T), origin='lower', extent=[-3, 3, -3, 3], cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Clifford Attractor Test', color='white', fontsize=16)
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Single frame test successful!")

# Run test first
print("🧪 Testing system...")
test_single_frame()

print("\n" + "="*60)
print("🎬 CLIFFORD ATTRACTOR VIDEO GENERATOR")
print("="*60)

print("\nChoose your video duration:")
print("1. Quick test (10 seconds)")  
print("2. Short video (30 seconds)")
print("3. Full video (60 seconds)")
print("4. Long video (120 seconds)")

print("\nUncomment one of these lines to generate:")
print("# create_fast_clifford_video(duration_seconds=10, fps=20)   # Quick test")
print("# create_fast_clifford_video(duration_seconds=30, fps=24)   # Short video") 
print("# create_fast_clifford_video(duration_seconds=60, fps=24)   # 1 minute video")
print("# create_fast_clifford_video(duration_seconds=120, fps=30)  # 2 minute video")

# Uncomment ONE line below to create your video:

# Quick 10-second test (recommended first)
#create_fast_clifford_video(duration_seconds=10, fps=20)

# Full 1-minute video (uncomment for final version)
create_fast_clifford_video(duration_seconds=60, fps=24)
