import json
import os
import urllib.request
import urllib.error
import glob

# --- Configuration ---
NUM_IMAGES_TO_DOWNLOAD = 50 

# Automatically find the RIO json file you just copied
json_files = glob.glob("dataset/rio/RIO_common_all*.json")
if not json_files:
    print("❌ Error: Could not find the RIO JSON file in dataset/rio/")
    exit()

JSON_PATH = json_files[0]
print(f"Reading annotations from: {JSON_PATH}")

with open(JSON_PATH, 'r') as f:
    dataset = json.load(f)

# Extract unique image IDs
unique_image_ids = set()
for ann in dataset:
    # Handle different possible JSON key formats for image ID
    img_id = ann.get('image_id') or ann.get('image')
    if img_id is not None:
        # Strip out any file extensions if they are included in the ID string
        if isinstance(img_id, str):
            img_id = img_id.split('.')[0]
        unique_image_ids.add(int(img_id))
        
    if len(unique_image_ids) >= NUM_IMAGES_TO_DOWNLOAD:
        break

print(f"Found {len(unique_image_ids)} unique images to fetch. Starting download...\n")

# Download the images directly from COCO servers
for idx, image_id in enumerate(unique_image_ids):
    # COCO uses 12-digit zero-padded numbers
    image_name_train = f"COCO_train2014_{image_id:012d}.jpg"
    image_name_val = f"COCO_val2014_{image_id:012d}.jpg"
    
    url_train = f"http://images.cocodataset.org/train2014/{image_name_train}"
    url_val = f"http://images.cocodataset.org/val2014/{image_name_val}"
    
    save_path_train = os.path.join("dataset/coco/train2014", image_name_train)
    save_path_val = os.path.join("dataset/coco/val2014", image_name_val)
    
    if os.path.exists(save_path_train) or os.path.exists(save_path_val):
        print(f"[{idx+1}/{NUM_IMAGES_TO_DOWNLOAD}] Image {image_id} already exists. Skipping.")
        continue

    print(f"[{idx+1}/{NUM_IMAGES_TO_DOWNLOAD}] Fetching Image {image_id}...")
    
    # Try Train split first
    try:
        urllib.request.urlretrieve(url_train, save_path_train)
    except urllib.error.HTTPError:
        # If not in Train, it must be in Val split
        try:
            urllib.request.urlretrieve(url_val, save_path_val)
        except Exception as e:
            print(f"  ❌ Failed to download {image_id} from both splits: {e}")

print("\n✅ Mini-dataset download complete! You can now run evaluate.py")