import sys
import types
import importlib.machinery

# Avoid hard dependency on flash_attn when running on CPU/Windows.
flash_attn_stub = types.ModuleType("flash_attn")
flash_attn_stub.__spec__ = importlib.machinery.ModuleSpec("flash_attn", None)

def _flash_attn_disabled(*_args, **_kwargs):
    raise RuntimeError("flash_attn is disabled (CPU mode)")

flash_attn_stub.flash_attn_func = _flash_attn_disabled
flash_attn_stub.flash_attn_kvpacked_func = _flash_attn_disabled
flash_attn_stub.flash_attn_qkvpacked_func = _flash_attn_disabled
sys.modules.setdefault("flash_attn", flash_attn_stub)

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image, ImageDraw
import time
import os
import config


class VisionEngine:
    def __init__(self):
        print("[INFO] Loading Florence-2 Vision Engine (copy)...")
        self.processor = AutoProcessor.from_pretrained(
            config.FLORENCE_MODEL_PATH, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config.FLORENCE_MODEL_PATH,
            torch_dtype=config.DTYPE,
            trust_remote_code=True
        ).to(config.DEVICE)
        print("[INFO] Vision Engine Ready (copy).")

    def analyze_scene(self, image: Image.Image):
        """Runs a two-stage vision query and returns (scene_str, labels, bboxes)."""
        start_time = time.time()

        task_1 = "<DENSE_REGION_CAPTION>"
        inputs_1 = self.processor(text=task_1, images=image, return_tensors="pt").to(config.DEVICE, config.DTYPE)

        gen_ids_1 = self.model.generate(
            input_ids=inputs_1["input_ids"],
            pixel_values=inputs_1["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3,
        )
        text_1 = self.processor.batch_decode(gen_ids_1, skip_special_tokens=False)[0]
        parsed_1 = self.processor.post_process_generation(text_1, task=task_1, image_size=image.size)

        labels = parsed_1[task_1].get("labels", [])
        bboxes = parsed_1[task_1].get("bboxes", [])

        spatial_graph = self._build_scene_graph(labels, bboxes, image.size)

        task_2 = "<VQA>"
        question = "Detail the materials (e.g., plastic, metal, glass), colors, and physical properties (e.g., fragile, insulated, sharp) of the objects in this image. Keep it short."

        inputs_2 = self.processor(text=task_2 + question, images=image, return_tensors="pt").to(config.DEVICE, config.DTYPE)

        gen_ids_2 = self.model.generate(
            input_ids=inputs_2["input_ids"],
            pixel_values=inputs_2["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3,
        )
        physical_details = self.processor.batch_decode(gen_ids_2, skip_special_tokens=False)[0]
        physical_details = physical_details.replace("<s>", "").replace("</s>", "").strip()

        merged_scene_context = f"""
[Spatial Locations]
{spatial_graph}

[Physical & Material Properties]
{physical_details}
        """.strip()

        print(f"⏱️ Vision Processing Time (2 Passes): {time.time() - start_time:.3f} sec")

        return merged_scene_context, labels, bboxes

    def _build_scene_graph(self, labels, bboxes, image_size):
        if not labels:
            return "No objects detected in the scene."

        img_width = image_size[0]
        scene_lines = []

        for i, (label, bbox) in enumerate(zip(labels, bboxes)):
            center_x = bbox[0] + (bbox[2] - bbox[0]) / 2

            if center_x < img_width * 0.33:
                location = "on the left"
            elif center_x > img_width * 0.66:
                location = "on the right"
            else:
                location = "in the center"

            scene_lines.append(f"- Object {i+1}: '{label}' located {location}.")

        return "\n".join(scene_lines)

    @staticmethod
    def draw_and_save_boxes(image: Image.Image, labels, bboxes, out_path: str):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        draw = ImageDraw.Draw(image)
        for label, bbox in zip(labels, bboxes):
            # bbox assumed [x1, y1, x2, y2]
            draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline="red", width=3)
            draw.text((bbox[0], max(bbox[1]-12, 0)), str(label), fill="red")

        image.save(out_path, format="JPEG")
