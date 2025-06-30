# -*- coding: utf-8 -*-
# author: haroldchen0414

from imutils import paths
import base64
import json
import os

def write_to_json(image_path, output_file="ukbench100.json"):
    imagePaths = list(paths.list_images(image_path))
    data = []

    for imagePath in imagePaths:
        with open(imagePath, "rb") as img:
            imageBase64 = base64.b64encode(img.read()).decode("utf-8")

        data.append({
            "filename": os.path.basename(imagePath),
            "image_base64": imageBase64,
            "size_kb": os.path.getsize(imagePath) / 1024
        })
        
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def load_image_from_json(json_path):
    os.makedirs("ukbench100", exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    for item in data:
        binaryData = base64.b64decode(item["image_base64"])

        with open(os.path.join("ukbench100", item["filename"]), "wb") as img:
            img.write(binaryData)

#write_to_json("ukbench100")
load_image_from_json("ukbench100.json")