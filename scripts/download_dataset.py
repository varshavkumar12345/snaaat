import os
import requests
import time
from tqdm import tqdm

FILES = [
    {
        "name": "NumFocus_Jan22-Dec24_GH_Activities.zip",
        "url": "https://zenodo.org/records/14230406/files/NumFocus_Jan22-Dec24_GH_Activities.zip",
        "expected_size": 236409310
    }
]

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)

def download_file(file_info):
    target_path = os.path.join(DATA_DIR, file_info["name"])
    url = file_info["url"]
    expected_size = file_info["expected_size"]

    initial_size = 0
    if os.path.exists(target_path):
        initial_size = os.path.getsize(target_path)
        if initial_size == expected_size:
            print(f"File {file_info['name']} already completely downloaded ({initial_size} bytes).")
            return target_path
        elif initial_size > expected_size:
            print(f"File size exceeds expected size, removing and restarting...")
            os.remove(target_path)
            initial_size = 0
        else:
            print(f"Resuming {file_info['name']} from {initial_size} / {expected_size} bytes...")

    headers = {"User-Agent": "Mozilla/5.0"}
    if initial_size > 0:
        headers["Range"] = f"bytes={initial_size}-"

    max_retries = 5
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=headers, stream=True, timeout=30)
            mode = "ab" if initial_size > 0 else "wb"
            if initial_size > 0 and r.status_code != 206:
                print("Server does not support partial content, restarting download...")
                mode = "wb"
                initial_size = 0

            with open(target_path, mode) as f:
                with tqdm(total=expected_size, initial=initial_size, unit='B', unit_scale=True, desc=file_info["name"]) as pbar:
                    for chunk in r.iter_content(chunk_size=1024 * 512):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            final_size = os.path.getsize(target_path)
            if final_size == expected_size:
                print(f"\nSuccessfully downloaded {file_info['name']}.")
                return target_path
            else:
                print(f"\nSize mismatch: got {final_size}, expected {expected_size}. Retrying...")
                initial_size = final_size
                headers["Range"] = f"bytes={initial_size}-"
        except Exception as e:
            print(f"\nError during download attempt {attempt+1}: {e}")
            time.sleep(3)
            if os.path.exists(target_path):
                initial_size = os.path.getsize(target_path)
                headers["Range"] = f"bytes={initial_size}-"

    raise RuntimeError(f"Failed to download {file_info['name']} after {max_retries} attempts.")

if __name__ == "__main__":
    for item in FILES:
        download_file(item)

