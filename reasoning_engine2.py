import time
import contextlib
from io import StringIO
from llama_cpp import Llama
import config
import re


class ReasoningEngine:
    def __init__(self):

        self.llm = Llama(
            model_path=config.LLM_MODEL_PATH,

            # ⚡ performance
            n_ctx=4096,
            n_threads=config.LLM_THREADS or 8,
            n_batch=512,

            # CPU safe
            n_gpu_layers=0,

            # ⚡ faster loading
            use_mmap=True,
            use_mlock=False,

            verbose=False
        )

        print("[INFO] Reasoning Engine Ready.")

    def generate_physics_prompt(self, task: str, scene_graph: str, labels=None) -> str:

        """Generate prompt with object indices for tracking which object is selected."""

        objects_list = ""

        if labels:

            objects_list = "Objects available:\n"

            for i, label in enumerate(labels, 1):

                objects_list += f"- Object {i}: {label}\n"

            objects_list += "\n"



        return f"""

You are a robotic reasoning engine.



Use ONLY physical properties:

- material

- shape

- size

- rigidity

- state



Do NOT hallucinate.



{objects_list}

Task:

{task}



Scene:

{scene_graph}

Think in this way: First, identify the task requirements based on the given task description. Then, analyze each object in the scene graph based on their physical properties and how they relate to the task requirements. Finally, rank the objects in terms of suitability for the task and select the best one.
1. Task Requirements:

- ...



2. Final Suitable Objects (Ranked):

- Object [X]: reason

- Object [Y]: reason

(If none: None found.)



3. Selected Object:
- Object [X]


Output format : 
Output as clearly structured text, with the selected object boxed like this: $\boxed{{X}}$ where X is the object number.
Reason : The reasoning behind the selection should be based on the physical properties of the objects and how they relate to the task requirements. For example, if the task is "I need to cut vegetables," and Object 1 is a knife while Object 2 is a spoon, the reasoning might be:

""".strip()



    def parse_selected_object(self, output_text: str, labels=None) -> int:
        """Extract the selected object number from the LLM output.
        
        Strategy:
        1) Look for boxed answer: $\boxed{N}$
        2) Look for explicit 'Selected Object: Object N' line
        3) Look for any 'Object N' mention
        4) Fallback: match object names mentioned in output to provided labels
        """
        # 1) Check for boxed answer format: $\boxed{N}$
        m = re.search(r'\$\\boxed\{(\d+)\}\$', output_text)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
        
        # 2) Look for explicit "Selected Object: Object N"
        m = re.search(r'Selected\s+Object\s*[:\-]?\s*Object\s*\[?(\d+)\]?', output_text, flags=re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
        
        # 3) Look for any "Object N" mention
        m = re.search(r'Object\s*\[?(\d+)\]?', output_text)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
        
        # 4) Fallback: match object names to labels
        if labels:
            out_lower = output_text.lower()
            for i, lbl in enumerate(labels):
                if not lbl:
                    continue
                if lbl.lower() in out_lower:
                    return i + 1
        
        return None

    def get_affordance(self, task: str, scene_graph: str, labels=None):
        """Get affordance reasoning and return the output text + selected object index."""
        prompt = self.generate_physics_prompt(task, scene_graph, labels)

        print("\n--- LLM Reasoning Output ---")
        start_time = time.time()

        stderr_buffer = StringIO()

        with contextlib.redirect_stderr(stderr_buffer):
            try:
                # ⚡ NON-stream (faster)
                response = self.llm.create_completion(
                    prompt=prompt,
                    max_tokens=800,
                    temperature=0.0,
                    stop=["</s>", "###"],
                    top_k=40,
                    top_p=0.95,
                    repeat_penalty=1.1
                )

                output_text = response["choices"][0]["text"]
                print(output_text)

            except Exception as e:
                print(f"[ERROR] LLM failed: {e}")
                output_text = ""

        print(f"\n⏱️ Reasoning Time: {time.time() - start_time:.3f} sec")

        # Extract which object was selected (pass labels for fallback name matching)
        selected_obj_idx = self.parse_selected_object(output_text, labels=labels)
        
        return {
            "reasoning": output_text,
            "selected_object_index": selected_obj_idx
        }