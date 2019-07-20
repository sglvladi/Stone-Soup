import numpy as np

from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
from skimage.feature import canny
from skimage.util import img_as_float, img_as_ubyte
from skimage.color import rgb2gray


def find_horizon_lines(image_rgb, theta=None, method='standard', mask=None,
                       **kwargs):
    """ Detect and return horizon lines in an image

    Parameters
    ----------
    image_rgb : (M, N, 3) ndarray
        Input image that is M pixels high and N pixels wide. The image is
        expected to be in RGB format.
    theta : 1D ndarray of double, optional
        Angles at which to compute the transform, in radians.
        Defaults to a vector of 180 angles evenly spaced from -pi/2 to pi/2.
    method : str, optional
        Method to be used to detect the lines. Options are 'standard' and
        'probabilistic' that correspond to the `standard and probabilistic
        Hough Transforms <https://scikit-image.org/docs/dev/auto_examples
        /edges/plot_line_hough_transform.html>`_. Default is 'standard'.
    mask : (M, N) ndarray of bool, optional
        Mask to be applied prior to edge detection. The mask should have the
        same size as the provided `image`, with zero values for pixels that
        should be masked out. Defaults to `np.ones((M,N))`, meaning that all
        pixels are used in the edge detection.

    Returns
    -------
    lines : list of tuples
        List of lines identified, lines in format ((x0, y0), (x1, y1)),
        indicating line start and end, as well as the peak.

    """
    # Line finding using the Probabilistic Hough Transform

    # Pre-process frame
    image = img_as_ubyte(rgb2gray(image_rgb)) # RGB to gray-scale
    num_rows, num_cols = image.shape

    # Edge detection
    edges = canny(image, 2, 1, 25, mask=mask)

    lines = []
    if method == 'standard':
        # Standard Hough transform
        h, a, d = hough_line(edges, theta=theta)
        peaks, angles, dists = hough_line_peaks(h, a, d, threshold=10)

        for peak, angle, dist in zip(*(peaks, angles, dists)):
            y0 = int((dist - 0 * np.cos(angle)) / np.sin(angle))
            y1 = int((dist - image.shape[1] * np.cos(angle)) / np.sin(angle))
            line = ((0, y0), (num_cols, y1))
            lines.append(line)
    else:
        # Probabilistice Hough Transform
        lines = probabilistic_hough_line(edges, threshold=10, line_length=5,
                                         line_gap=10, theta=theta)

    return lines

