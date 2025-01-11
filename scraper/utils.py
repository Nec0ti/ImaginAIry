import os
import requests

def download_image(url, save_path):
    """
    Downloads an image from a URL and saves it to the specified file path.

    Parameters:
    - url: URL of the image to download
    - save_path: The file path where the image will be saved
    """
    try:
        img_data = requests.get(url).content
        with open(save_path, "wb") as f:
            f.write(img_data)
        print(f"Image saved at {save_path}")
    except Exception as e:
        print(f"Error downloading image: {e}")

