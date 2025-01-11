import os
import requests
from bs4 import BeautifulSoup
from utils import download_image
from config import SEARCH_TERMS, IMAGE_COUNT, SAVE_DIR

def scrape_images(search_term, save_dir, image_count=100):
    """
    Scrapes images based on a search term and saves them to the specified directory.

    Parameters:
    - search_term: The search term to look for on Google Images
    - save_dir: Directory where images will be saved
    - image_count: Number of images to scrape
    """

    # Prepare the directory for saving images
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    url = f"https://www.google.com/search?hl=en&tbm=isch&q={search_term}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    # Get the HTML page from Google Images
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    # Find all the image elements
    img_tags = soup.find_all("img", attrs={"src": True})

    # Download images
    count = 0
    for img_tag in img_tags:
        if count >= image_count:
            break
        img_url = img_tag["src"]
        download_image(img_url, os.path.join(save_dir, f"{search_term}_{count}.jpg"))
        count += 1

if __name__ == "__main__":
    for search_term in SEARCH_TERMS:
        scrape_images(search_term, os.path.join(SAVE_DIR, search_term), IMAGE_COUNT)
