import os
import time
from llama_cpp import Llama
from io import StringIO
import contextlib
from datetime import datetime
from huggingface_hub import hf_hub_download

# ----------------------------------------------------
# DYNAMIC MODEL PATH - LLAMA 3.2 1B
# ----------------------------------------------------
MODEL_DIR = "models/llm"
MODEL_FILENAME = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

if not os.path.exists(MODEL_PATH):
    print(f"\n[INFO] Downloading Llama 3.2 1B model to {MODEL_PATH}...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    hf_hub_download(
        repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF", 
        filename=MODEL_FILENAME, 
        local_dir=MODEL_DIR
    )

start_load = time.time()
print("[INFO] Loading Llama-3.2-1B Reasoning Engine...")
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=4,            
    n_gpu_layers=0,         
    use_mlock=True,
    verbose=False           
)
end_load = time.time()

def generate_cot_prompt(task, attribute_list):
    formatted_scene = "\n".join(attribute_list) if isinstance(attribute_list, list) else str(attribute_list)

    return f"""System: You are a generalized Robotic Affordance AI. Your core function is to deduce the physical requirements of ANY task from first principles.

STRICT REASONING RULES:
1. Material Hierarchy: Metallic objects (Steel/Silver/Iron) ALWAYS possess higher structural integrity and sharpness-potential than Plastic or Ceramic objects.
2. Geometry Priority: For tasks involving "opening," "chopping," or "cutting," you MUST prioritize objects with a "blade," "edge," or "pointed tip."
3. Primary vs Secondary: Evaluate the business-end of the tool (the blade/tip) before the handle. A handle shape NEVER invalidates a functional blade.
4. No "None" Policy: You are a decision-making engine. "None" is not an option. Pick the object that possesses the highest percentage of required physical traits.
5. Absolute Tie-Breaker: If Object 1 is a Knife and Object 2 is a Mug, and the task is "opening/cutting," the Knife wins 100% of the time.

### EXAMPLE INTERACTION
Task: "cut an apple"
Scene:
- Object 1: A metal spoon with a rounded edge.
- Object 2: A steel steak knife with a serrated blade.
- Object 3: A plastic bowl.

Output:
**1. Task Mechanics:**
Cutting requires a material harder than the target (apple) and a thin edge geometry to concentrate force and shear fibers.

**2. Object Analysis:**
- Object 1 (spoon): Metal (Hard), but edge is rounded. (PARTIAL PASS - Hard but poor geometry)
- Object 2 (knife): Steel (Hard), has a serrated blade and thin edge. (PERFECT PASS - Hard with ideal shearing geometry)
- Object 3 (bowl): Plastic (Soft), no edge. (FAILS - Lacks material hardness and geometry)

**3. Final Suitable Object:**
Object 2 is the superior choice due to its steel construction and specialized cutting geometry.
WINNER: Object 2
### END EXAMPLE

Task: "{task}"
Scene:
{formatted_scene}

Output:
**1. Task Mechanics:**
"""

def get_affordance_reasoning(task_name, attribute_list):
    prompt = generate_cot_prompt(task_name, attribute_list)
    
    print("\n" + "="*50)
    print("📥 [DEBUG] EXACT INPUT PROMPT SENT TO LLM:")
    print("="*50)
    print(prompt)
    print("="*50 + "\n")

    output = "**1. Task Mechanics:**\n"
    first_token_ts = None
    last_token_ts = None
    llm_start = time.time()

    print("📤 [DEBUG] LLM OUTPUT STREAMING START:")
    print(output, end="") 
    
    stderr_buffer = StringIO()
    with contextlib.redirect_stderr(stderr_buffer):
        stream = llm.create_completion(
            prompt=prompt,
            max_tokens=700,   
            temperature=0.0, # Keeps the model strictly deterministic
            repeat_penalty=1.3, # Penalize repeated tokens to prevent looping
            frequency_penalty=0.2, # Further discourage repetition
            stop=["### END", "<|eot_id|>", "<|end_of_text|>"], 
            stream=True
        )
        
        recent_chunks = []  # Track recent output for repetition detection
        for chunk in stream:
            now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            if first_token_ts is None:
                first_token_ts = now_ts
                print(f"\n🟢 FIRST TOKEN TS: {first_token_ts}\n")
            
            last_token_ts = now_ts
            token_text = chunk["choices"][0].get("text", "")
            print(token_text, end="", flush=True)
            output += token_text

            # Repetition detection: stop if a long phrase repeats
            recent_chunks.append(token_text)
            if len(recent_chunks) > 20:
                recent_text = "".join(recent_chunks[-40:])
                half = len(recent_text) // 2
                if half > 30 and recent_text[:half].strip() and recent_text[:half].strip() in recent_text[half:]:
                    print("\n⚠️ Repetition loop detected, stopping generation early.")
                    break
            
        print(f"\n\n🔴 LAST TOKEN TS:  {last_token_ts}")

    llm_end = time.time()
    
    reasoning_time = llm_end - llm_start
    print(f"⏱️ Llama 3.2 Reasoning Time: {reasoning_time:.2f} sec")
    
    return output.strip(), {}

