import time

import contextlib

from io import StringIO

from llama_cpp import Llama

import config

import re





class ReasoningEngine:

    def __init__(self):

        print("[INFO] Loading LLM Reasoning Engine...")



        self.llm = Llama(

            model_path=config.LLM_MODEL_PATH,



            # ⚡ performance

            n_ctx=1024,

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



OUTPUT FORMAT:



1. Task Requirements:

- ...



2. Final Suitable Objects (Ranked):

- Object [X]: reason

- Object [Y]: reason

(If none: None found.)



3. Selected Object:

- Object [X]



4. Unsuitable Objects:

- Object [Z]: reason

""".strip()



    def parse_selected_object(self, output_text: str) -> int:

        """Extract the selected object number from the LLM output."""

        # Look for patterns like "Object 1", "Object [1]", "1:", etc. in the Selected Object section

        lines = output_text.split('\n')

        selected_idx = None

       

        in_selected_section = False

        for line in lines:

            if "selected object" in line.lower():

                in_selected_section = True

                continue

           

            if in_selected_section and ("unsuitable" in line.lower() or "task requirement" in line.lower()):

                break

           

            if in_selected_section:

                # Look for Object X pattern

                match = re.search(r'Object\s*\[?(\d+)\]?', line)

                if match:

                    selected_idx = int(match.group(1))

                    break

       

        # If not found in Selected Object section, look for the top-ranked suitable object

        if selected_idx is None:

            match = re.search(r'Object\s*\[?(\d+)\]?', output_text)

            if match:

                selected_idx = int(match.group(1))

       

        return selected_idx



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

                    max_tokens=300,

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



        # Extract which object was selected

        selected_obj_idx = self.parse_selected_object(output_text)

       

        return {

            "reasoning": output_text,

            "selected_object_index": selected_obj_idx

        }