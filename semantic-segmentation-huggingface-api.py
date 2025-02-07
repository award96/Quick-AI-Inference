import requests
import json
import base64
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.patches import Patch
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# --- API call setup ---
API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/nvidia/segformer-b0-finetuned-ade-512-512"

headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "x-wait-for-model": "true",
    "Content-Type": "application/octet-stream"
}

palette = [
    (230, 25, 75),   # red
    (60, 180, 75),   # green
    (255, 225, 25),  # yellow
    (0, 130, 200),   # blue
    (245, 130, 48),  # orange
    (145, 30, 180),  # purple
    (70, 240, 240),  # cyan
    (240, 50, 230),  # magenta
    (210, 245, 60),  # lime
    (250, 190, 190), # pink
]

def query_image(image_path):
    """Reads the image file and sends it to the Hugging Face API.
    Returns the JSON response with segmentation results."""
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    response = requests.post(API_URL, headers=headers, data=image_bytes)
    if response.status_code != 200:
        raise Exception(f"Request failed: {response.status_code}, {response.text}")
    return response.json()

def plot_segmentation_map(original_image, segmentation_results, output_path="segmentation_map.png"):
    """Overlays segmentation masks on the original image and adds a legend for the class labels.
    
    The function creates an empty overlay, assigns a random RGB color to each unique class (using
    the 'label' from the segmentation result), and then for each result it creates a binary mask 
    (thresholding at 128). Finally, it alpha-blends the overlay with the original image and plots 
    a legend showing each label with its assigned color."""
    
    orig_np = np.array(original_image)
    height, width = orig_np.shape[:2]
    overlay = np.zeros((height, width, 3), dtype=np.uint8)
    label_to_color = {}
    
    for result in segmentation_results:
        label = result.get("label", "unknown")
        mask_data = result.get("mask")
        
        # Decode the mask if necessary.
        if isinstance(mask_data, str):
            mask_bytes = base64.b64decode(mask_data)
            mask_image = Image.open(io.BytesIO(mask_bytes))
        elif isinstance(mask_data, Image.Image):
            mask_image = mask_data
        else:
            print(f"Unexpected mask type for {label}: {type(mask_data)}")
            continue
        
        mask_np = np.array(mask_image)
        binary_mask = mask_np > 128  # simple threshold to create a binary mask
        
        # Assign a color from the palette if not already set for this label.
        if label not in label_to_color:
            # Choose the next color from the palette, cycling if needed.
            palette_index = len(label_to_color) % len(palette)
            label_to_color[label] = palette[palette_index]
            
        color = np.array(label_to_color[label], dtype=np.uint8)
        overlay[binary_mask] = color

    # Blend the original image and the overlay using alpha blending
    alpha = 0.5
    blended = (alpha * orig_np + (1 - alpha) * overlay).astype(np.uint8)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(blended)
    plt.axis("off")
    plt.title("Semantic Segmentation Map")
    
    # Create legend patches for each class label
    patches = [Patch(facecolor=np.array(color)/255, label=label) for label, color in label_to_color.items()]
    plt.legend(handles=patches, loc="upper right")
    plt.show()
    
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Segmentation map saved as {output_path}")

# --- Main execution ---
if __name__ == "__main__":
    image_path = "image-of-road.jpg"
    
    original_image = Image.open(image_path)
    try:
        segmentation_results = query_image(image_path)
        print("API Response:")
        print(json.dumps(segmentation_results, indent=2))
    except Exception as e:
        print(e)
        exit(1)
    
    plot_segmentation_map(original_image, segmentation_results, output_path="segmentation_map.png")
    print("Done")
