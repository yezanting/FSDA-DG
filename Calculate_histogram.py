import numpy as np

def convert_image_data(image_data):
    """
    Convert single-channel image data by preserving pixels with value 0 and converting pixels with value greater than 0 to 1.
    
    Parameters:
    image_data (numpy.ndarray): The single-channel image data to be converted.
    
    Returns:
    numpy.ndarray: The converted image data.
    """
    # Using numpy where function to perform the conversion
    converted_data = np.where(image_data > 0, 1, 0)
    return converted_data


