import sys
import types

flash_attn_stub = types.ModuleType("flash_attn")

def dummy_func(*args, **kwargs):
    raise RuntimeError("flash_attn is disabled (CPU mode)")

# Stub common functions used internally
flash_attn_stub.flash_attn_func = dummy_func
flash_attn_stub.flash_attn_kvpacked_func = dummy_func
flash_attn_stub.flash_attn_qkvpacked_func = dummy_func

sys.modules["flash_attn"] = flash_attn_stub
import os
import time
from datetime import datetime
import cv2
from PIL import Image, ImageDraw
from boltons.fileutils import mkdir_p
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from COT import get_affordance_reasoning

torch.set_grad_enabled(False)


taskid2name = {
    # Image 1 (laptop, notebook, pen, stapler)
    1: "I need to permanently attach these loose printed reports.",
    2: "I want to write down some notes for my upcoming meeting.",
    3: "I need to type and edit a document efficiently.",

    # Image 2 (knife, cutting board, peeler)
    4: "I need a safe, flat surface to chop vegetables.",
    5: "I want to peel the skin off fruits or vegetables easily.",
    6: "I need to slice ingredients cleanly for cooking.",

    # Image 3 (pan, spatula, tongs)
    7: "I need to flip a hot roti without scratching the pan.",
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
    27: "I need a flat surface to serve my meal."
}


SAVING_DIRECTORY = "./results_live"

class FlorenceLiveReasoning:
    def __init__(self, task_number: int):
        self.task_name = taskid2name.get(task_number, "generic task")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.florence_path = "models/Florence-2-base-ft"

        self.det_model = AutoModelForCausalLM.from_pretrained(
            self.florence_path,
            torch_dtype=self.dtype,
            trust_remote_code=True,   # needed for Florence
            local_files_only=True
        ).to(self.device)
        self.det_model.config.attn_config = {"attn_impl": "eager"}

        self.processor = AutoProcessor.from_pretrained(
            self.florence_path
        )

        self.base_out_dir = os.path.join(SAVING_DIRECTORY, f"task_{task_number}")
        self.green_dir = os.path.join(self.base_out_dir, "florence_detections")
        mkdir_p(self.green_dir)

    def process_frame(self, frame, frame_id):
        start = time.time()
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # ---------- PASS 1 ----------
        task = "<OD>"
        inputs = self.processor(text=task, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            ids = self.det_model.generate(
                **inputs,
                max_new_tokens=512,
                num_beams=1
            )

        parsed = self.processor.post_process_generation(
            self.processor.batch_decode(ids)[0],
            task=task,
            image_size=image.size
        )

        bboxes = parsed[task].get("bboxes", [])

        # remove duplicates
        unique = []
        for box in bboxes:
            if not any(abs(box[0]-u[0]) < 30 for u in unique):
                unique.append(box)

        # ---------- PASS 2 ----------
        MAX_OBJECTS = 5   # 🔥 speed gain

        descs = []

        for i, (x1, y1, x2, y2) in enumerate(unique[:MAX_OBJECTS]):
            crop = image.crop((x1, y1, x2, y2))

            inputs = self.processor(
                text="<MORE_DETAILED_CAPTION>",
                images=crop,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                ids = self.det_model.generate(
                    **inputs,
                    max_new_tokens=64,
                    num_beams=1
                )

            text = self.processor.batch_decode(ids)[0]
            desc = text.replace("<s>", "").replace("</s>", "").strip()
            descs.append(f"- Object {i+1}: {desc}")

        if not descs:
            return frame, [], ""

        reasoning_text, _ = get_affordance_reasoning(self.task_name, descs)

        print(f"\n🚀 Total Time: {time.time() - start:.2f}s")

        return frame, unique[:MAX_OBJECTS], reasoning_text
    














































