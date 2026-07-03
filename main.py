# main.py
import sys
sys.stdout.reconfigure(encoding='utf-8')

import argparse
from PIL import Image
import os
from datetime import datetime

# Prefer using copied engines if available so originals remain unchanged.
try:
    from vision_engine2 import VisionEngine
except Exception:
    from vision_engine import VisionEngine

try:
    from reasoning_engine_3 import ReasoningEngine
except (ImportError, ModuleNotFoundError):
    try:
        from reasoning_engine2 import ReasoningEngine
    except Exception:
        from reasoning_engine import ReasoningEngine

        
TASK_ID_TO_NAME = {
    # Image 1 (laptop, notebook, pen, stapler)
    1: "I need to permanently attach these loose printed reports.",
    2: "I want to write down some notes for my upcoming meeting.",
    3: "I need to type and edit a document efficiently.",

    # Image 2 (knife, cutting board, peeler)
    4: "I need a safe, flat surface to chop vegetables.",
    5: "I want to peel the skin off fruits or vegetables easily.",
    6: "I need to slice ingredients cleanly for cooking.",

    # Image 3 (pan, spatula, tongs)
    7: "I need to flip a hot roti scratching the pan.",
    8: "I want to safely grab and turn hot food while cooking.",
    9: "I need to cook food evenly on a heated surface.",

    # Image 4 (mug, glass, water bottle)
    10: "I need to safely hold boiling hot tea without burning my fingers.",
    11: "I want to drink water conveniently while moving around.",
    12: "I need a container to pour and drink liquids.",

    # Image 5 (badminton racket, shuttle, towel)
    13: "I want to hit a lightweight projectile over a net.",
    14: "I need to wipe sweat during a sports activity.",
    15: "I want to practice hitting a shuttlecock during a game.",

    # Image 6 (scissors, tape, measuring tape)
    16: "I need to cut open a thick cardboard package.",
    17: "I want to seal or stick objects together securely.",
    18: "I need to measure the length of an object accurately.",

    19: "I need to cook food.",
    20: "I need to drink cold coffee.",
    21: "I want to cut an apple",

    22: "I need to carry water for long trip.",
    23: "I want to carry hot water for long trip.",
    24: "I need to hammer a nail into the wall.",

    # Image 9 (plate, spoon, fork)
    25: "I need to eat a bowl of liquid soup.",
    26: "I want to eat solid food using proper utensils.",
    27: "I need a flat surface to serve my meal.",
    28: "I need to write notes on a piece of paper.",
    29: "i need to place a hot pan on the table without damaging it."
}

def _parse_args():
    parser = argparse.ArgumentParser(description="Run affordance reasoning on an image.")
    parser.add_argument("--image-path", required=True, help="Path to the input image.")
    parser.add_argument("--task", help="Free-form task prompt.")
    parser.add_argument("--task-number", type=int, help="Task ID (1-28).")
    return parser.parse_args()

def _resolve_task(task_text, task_number):
    if task_text and task_text.strip():
        return task_text.strip()
    if task_number in TASK_ID_TO_NAME:
        return TASK_ID_TO_NAME[task_number]
    return None

def main():
    args = _parse_args()
    user_task = _resolve_task(args.task, args.task_number)
    if not user_task:
        raise SystemExit("Provide --task or a valid --task-number (1-28).")

    # Initialize Engines
    vision = VisionEngine()
    reasoner = ReasoningEngine()

    print("\n📸 Processing image...")
    image = Image.open(args.image_path).convert("RGB")

    scene_result = vision.analyze_scene(image)

    # If the vision engine returned (scene_str, labels, bboxes) use that,
    # otherwise treat it as the original single string.
    labels = []
    bboxes = []
    if isinstance(scene_result, tuple) and len(scene_result) == 3:
        scene_graph, labels, bboxes = scene_result
    else:
        scene_graph = scene_result

    print(f"\n👁️ Vision System Detected:\n{scene_graph}")

    def clean_scene(scene_graph: str):
        lines = scene_graph.split("\n")
        clean_lines = []

        for line in lines:
            # remove Florence garbage lines
            if "QA>" in line or "<loc_" in line:
                continue
            if line.strip():
                clean_lines.append(line)

        return "\n".join(clean_lines)
    
    cleaned_scene = clean_scene(scene_graph)
    
    # Get affordance reasoning - now returns dict with reasoning and selected object
    affordance_result = reasoner.get_affordance(user_task, cleaned_scene, labels=labels)
    
    # Extract the selected object index (1-indexed)
    selected_obj_idx = None
    if isinstance(affordance_result, dict):
        selected_obj_idx = affordance_result.get("selected_object_index")
    
    # Draw bounding box ONLY for the selected object
    if labels and bboxes and selected_obj_idx is not None:
        # Convert 1-indexed to 0-indexed
        obj_idx = selected_obj_idx - 1
        if 0 <= obj_idx < len(labels):
            out_dir = os.path.join("results_image")
            os.makedirs(out_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(args.image_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(out_dir, f"{base}_selected_{timestamp}.jpg")

            # Draw only the selected object
            from PIL import ImageDraw
            boxed_img = image.copy()
            draw = ImageDraw.Draw(boxed_img)
            
            label = labels[obj_idx]
            bbox = bboxes[obj_idx]
            draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline="green", width=4)
            draw.text((bbox[0], max(bbox[1]-12, 0)), str(label), fill="green")
            
            boxed_img.save(out_path, format="JPEG")
            print(f"\n✅ Saved selected object image to: {out_path}")
            print(f"   Selected: Object {selected_obj_idx} ({label})")
    elif labels and bboxes:
        # Fallback: draw all if no selection made
        print(f"\n⚠️  Could not determine selected object, showing all detections...")
        out_dir = os.path.join("results_image")
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(args.image_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(out_dir, f"{base}_boxed_{timestamp}.jpg")

        draw_func = getattr(vision, "draw_and_save_boxes", None)
        if callable(draw_func):
            boxed_img = image.copy()
            draw_func(boxed_img, labels, bboxes, out_path)
        else:
            from PIL import ImageDraw
            boxed_img = image.copy()
            draw = ImageDraw.Draw(boxed_img)
            for label, bbox in zip(labels, bboxes):
                draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline="red", width=3)
                draw.text((bbox[0], max(bbox[1]-12, 0)), str(label), fill="red")
            boxed_img.save(out_path, format="JPEG")

        print(f"\n🖼️ Saved boxed image to: {out_path}")

if __name__ == "__main__":
    main()