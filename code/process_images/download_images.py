import os
import subprocess
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd

def download_single_image(image_link, save_path, retries=3, delay=1):
    """Download a single image using wget with retries."""
    for attempt in range(retries):
        try:
            cmd = [
                "wget", "-q", "--timeout=10", "--tries=1", "-O", save_path, image_link
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0 and os.path.exists(save_path):
                return True
        except Exception:
            pass
        time.sleep(delay)
    return False


def download_images_fast(df, image_folder, max_workers=16):
    """
    Downloads all images in df['image_link'] and saves them in image_folder.
    Uses multiple threads for speed.
    """
    os.makedirs(image_folder, exist_ok=True)
    download_tasks = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for _, row in df.iterrows():
            sample_id = str(row['sample_id'])
            image_link = row['image_link']

            if not isinstance(image_link, str) or not image_link.strip():
                continue

            suffix = Path(image_link).suffix
            if not suffix or len(suffix) > 5:
                suffix = ".jpg"

            save_path = os.path.join(image_folder, f"{sample_id}{suffix}")
            if os.path.exists(save_path):
                continue

            download_tasks.append(executor.submit(download_single_image, image_link, save_path))

        for f in tqdm(as_completed(download_tasks), total=len(download_tasks), desc="Downloading images"):
            _ = f.result()

    print(f"\n✅ All downloads complete. Images saved in: {image_folder}")


# Example usage:
if __name__ == "__main__":
    DATASET_FILE = '/home/user3/amazon/test_split_final.csv'
    test = pd.read_csv(DATASET_FILE)
    IMAGE_SAVE_FOLDER = "./test_images_fast/"

    download_images_fast(test, IMAGE_SAVE_FOLDER, max_workers=32)
