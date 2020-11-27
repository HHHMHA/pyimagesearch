import argparse
import requests
import time
import os

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="Path to output directory of images")
ap.add_argument("-n", "--num-images", type=int, default=500, help="# of images to download")
args = vars(ap.parse_args())

URL = "https://www.e-zpassny.com/vector/jcaptcha.do"
total_downloaded_images = 0

for i in range(args["num_images"]):
    try:
        response = requests.get(URL, timeout=100)
        save_path = os.path.sep.join([args["output"], f"{str(total_downloaded_images).zfill(5)}.jpg"])
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"[INFO] downloaded: {save_path}")
        total_downloaded_images += 1
    except:
        print("[INFO] error downloading image...")

    time.sleep(0.1)
