import os
os.environ["USE_FLASH_ATTENTION"] = "0"


import json
import os
import sys
import types
flash_attn_stub = types.ModuleType("flash_attn")
import importlib.machinery

flash_attn_stub.__spec__ = importlib.machinery.ModuleSpec(
    name="flash_attn",
    loader=None
)
def dummy_func(*args, **kwargs):
    raise RuntimeError("flash_attn is disabled (CPU mode)")

# Stub common functions used internally
flash_attn_stub.flash_attn_func = dummy_func
flash_attn_stub.flash_attn_kvpacked_func = dummy_func
flash_attn_stub.flash_attn_qkvpacked_func = dummy_func

sys.modules["flash_attn"] = flash_attn_stub
import cv2
import re
import glob
import csv
from florence_detection_test import FlorenceLiveReasoning

def calculate_iou(boxA, boxB):
    """Calculates Intersection over Union (IoU) for two boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    denominator = float(boxAArea + boxBArea - interArea)
    if denominator == 0:
        return 0.0
    return interArea / denominator

def extract_winning_object_id(reasoning_text):
    """Parses the LLM output to find the selected Object ID using robust regex."""
    
    # 1. Primary Check: Matches "WINNER: Object 2", "**WINNER**: 2", "WINNER: #2", etc.
    match = re.search(r"WINNER[^\d]*(\d+)", reasoning_text, re.IGNORECASE)
    if match:
        return int(match.group(1)) - 1 
        
    # 2. Fallback Check: If the model completely forgot the word "WINNER", 
    # we scan the final sentences to find the last mentioned "Object X"
    fallback_matches = re.findall(r"Object\s*(\d+)", reasoning_text, re.IGNORECASE)
    if fallback_matches:
        # Grab the very last object number it was talking about
        return int(fallback_matches[-1]) - 1
        
    return -1

def evaluate_rio_dataset():
    # 1. Automatically find the RIO JSON file
    json_files = glob.glob("dataset/rio/RIO_common_all*.json")
    if not json_files:
        print("❌ Error: Could not find the RIO JSON file in dataset/rio/")
        return
    annotations_file = json_files[0]

    # Directories where images might be stored
    train_dir = 'dataset/coco/train2014/'
    val_dir = 'dataset/coco/val2014/'

    with open(annotations_file, 'r') as f:
        dataset = json.load(f)
        
    pipeline = FlorenceLiveReasoning() 
    
    total_iou = 0
    evaluated_count = 0 
    correct_count = 0
    IOU_THRESHOLD = 0.5 
    
    # --- NEW: Set maximum test cases here ---
    MAX_TEST_CASES = 30
    
    csv_filename = "evaluation_results.csv"
    
    print(f"\n[EVALUATION] Starting test. Will run for maximum {MAX_TEST_CASES} cases.")
    print(f"[EVALUATION] Results will be saved to {csv_filename}...")
    
    # Open CSV file for writing
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ['Image ID', 'Intention', 'Expected Object (GT BBoxes)', 'Code Output (Pred BBox)', 'IoU', 'Correct (IoU >= 0.5)']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
    
        for idx, ann in enumerate(dataset):
            img_id_raw = ann.get('image_id') or ann.get('image')
            if not img_id_raw:
                continue
                
            if isinstance(img_id_raw, str):
                img_id_raw = img_id_raw.split('.')[0]
            image_id = int(img_id_raw)
            
            # Check both train and val folders for the image
            image_name_train = f"COCO_train2014_{image_id:012d}.jpg"
            image_name_val = f"COCO_val2014_{image_id:012d}.jpg"
            
            path_train = os.path.join(train_dir, image_name_train)
            path_val = os.path.join(val_dir, image_name_val)
            
            image_path = None
            if os.path.exists(path_train):
                image_path = path_train
            elif os.path.exists(path_val):
                image_path = path_val
                
            # If the image isn't downloaded, silently skip
            if not image_path:
                continue
                
            frame = cv2.imread(image_path)
            if frame is None:
                continue

            # Extract Task/Intention
            task_query = ann.get('expressions')
            if not task_query and 'sentences' in ann:
                task_query = [s['raw'] for s in ann['sentences']]
            
            if isinstance(task_query, list):
                task_query = task_query[0]
                
            # Extract Ground Truth Bounding Box(es)
            gt_bboxes = ann.get('bbox_list', [])
            if not gt_bboxes and 'bbox' in ann:
                gt_bboxes = [ann['bbox']]

            evaluated_count += 1
            print(f"\n--- Testing Sample {evaluated_count}/{MAX_TEST_CASES} (Image ID: {image_id}) ---")
            print(f"Intention: '{task_query}'")
            def clean_task(text):
                text = text.lower()
                text = re.sub(r'[^a-z\s]', '', text)
                return text.strip()
            
            task_query = clean_task(task_query)
            pipeline.task_name = task_query

            _, bboxes, reasoning = pipeline.process_frame(frame, frame_id=idx)
            
            best_iou = 0.0
            pred_box_str = "None"
            is_correct = "No"
            
            if not bboxes:
                print("❌ Pipeline found no objects. Score: 0.0")
            else:
                winning_index = extract_winning_object_id(reasoning)
                
                if 0 <= winning_index < len(bboxes):
                    pred_box = bboxes[winning_index]
                    pred_box_str = str([round(x, 1) for x in pred_box])
                    
                    # Find the max IoU against any valid ground-truth object
                    for gt_bbox in gt_bboxes:
                        gt_box_converted = [gt_bbox[0], gt_bbox[1], gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]]
                        iou = calculate_iou(gt_box_converted, pred_box)
                        if iou > best_iou:
                            best_iou = iou
                            
                    total_iou += best_iou
                    
                    if best_iou >= IOU_THRESHOLD:
                        is_correct = "Yes"
                        correct_count += 1
                        print(f"✅ Selected Object {winning_index + 1}. Max IoU: {best_iou:.4f} (Correct)")
                    else:
                        print(f"❌ Selected Object {winning_index + 1}. Max IoU: {best_iou:.4f} (Incorrect)")
                else:
                    print("❌ LLM failed to explicitly select a valid object ID. Score: 0.0")
            
            # Format Ground Truth boxes for the CSV
            gt_boxes_str = str([[round(x, 1) for x in box] for box in gt_bboxes])
            
            # Write row to CSV
            writer.writerow({
                'Image ID': image_id,
                'Intention': task_query,
                'Expected Object (GT BBoxes)': gt_boxes_str,
                'Code Output (Pred BBox)': pred_box_str,
                'IoU': round(best_iou, 4),
                'Correct (IoU >= 0.5)': is_correct
            })
            
            # --- NEW: Break the loop if we hit the limit ---
            if evaluated_count >= MAX_TEST_CASES:
                print(f"\n⏹️ Reached maximum test cases ({MAX_TEST_CASES}). Stopping evaluation.")
                break
            
    if evaluated_count > 0:
        mean_iou = total_iou / evaluated_count
        accuracy = (correct_count / evaluated_count) * 100
        print("\n" + "="*50)
        print(f"🏆 EVALUATION COMPLETE | Tested {evaluated_count} images")
        print(f"📈 Mean IoU: {mean_iou:.4f}")
        print(f"🎯 Accuracy (IoU >= {IOU_THRESHOLD}): {accuracy:.2f}% ({correct_count}/{evaluated_count})")
        print(f"📁 Results saved to: {os.path.abspath(csv_filename)}")
        print("="*50)
    else:
        print("\n⚠️ No valid images were found in your dataset directories to test.")

if __name__ == "__main__":
    evaluate_rio_dataset()