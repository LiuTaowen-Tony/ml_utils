import numpy as np
import matplotlib

def fig_to_array(fig: matplotlib.figure.Figure) -> np.ndarray:
    """
    Convert a Matplotlib figure to a numpy array.
    
    Args:
        fig (matplotlib.figure.Figure): The figure to convert
        
    Returns:
        numpy.ndarray: RGB array of shape (height, width, 3)
    """
    # Draw the figure first if it hasn't been drawn yet
    fig.canvas.draw()
    
    # Get the RGB buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (h, w, 3)
    
    return buf