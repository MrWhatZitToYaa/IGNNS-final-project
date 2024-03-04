import numpy as np


def get_gaussian_kernel(sigma: float, window_size: int = None) -> np.ndarray:
    if window_size is None:
        window_size = int(sigma * 3.5)

    return np.exp(-np.arange(-window_size, window_size + 1)**2 / (2 * sigma**2))


def partconv1d(data: np.ndarray, kernel: np.ndarray, periodic: bool = False) -> np.ndarray:
    '''
    Convolve data with a kernel, handling edges with appropriately normalized truncated kernel,
    optionally using periodic padding.

    Parameters
    ----------
    data : np.ndarray
        Data to be convolved.
    kernel : np.ndarray
        Kernel for convolution.
    periodic : bool, optional
        If True, the data is treated as circular and padded accordingly. Default is False.

    Returns
    -------
    np.ndarray
        Convolved data.
    '''
    if not isinstance(data, np.ndarray) or not isinstance(kernel, np.ndarray):
        raise ValueError("Data and kernel must be numpy arrays.")

    if len(kernel) % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    window_size = len(kernel) // 2

    if periodic:
        # Extend the data on both sides for circular data
        data = np.concatenate((data[-window_size:], data, data[:window_size]))

    # Convolve the middle section of the data with the kernel, i.e. where the data and the kernel overlap completely
    data_convolved_middle = np.convolve(data, kernel / kernel.sum(), mode='valid')

    # Convolve the edges of the data with the kernel, i.e. where the data and the kernel overlap partially
    data_convolved_left = np.empty(2 * window_size - 1)
    data_convolved_right = np.empty(2 * window_size - 1)
    for i in range(1, 2 * window_size):
        data_convolved_left[i - 1] = data[:i] @ kernel[-i:] / kernel[-i:].sum()
        data_convolved_right[i - 1] = data[- 2 * window_size + i:] @ kernel[:2 * window_size - i] / kernel[:2 * window_size - i].sum()

    # Convolve the data with the kernel
    data_convolved = np.concatenate((data_convolved_left[window_size - 1:], data_convolved_middle, data_convolved_right[:window_size]))

    if periodic:
        # Cut off the excess data
        data_convolved = data_convolved[window_size:-window_size]

    return data_convolved
