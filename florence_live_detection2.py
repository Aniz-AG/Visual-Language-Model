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
    1: "step on", 2: "open a amazon parcel", 3: "eat food", 4: "drink water",
    5: "play music", 6: "chop vegetables", 7: "type message", 8: "cook food",
    9: "clean floor", 10: "write text", 11: "measure length", 12: "read book",
    13: "paint picture", 14: "fix screw",
    15: "store warm water for long duration safely"
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
    














































