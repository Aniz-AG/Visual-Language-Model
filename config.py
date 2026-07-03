# config.py
import torch

# --- MODEL PATHS ---
FLORENCE_MODEL_PATH = "./models/Florence-2-base"
LLM_MODEL_PATH = "./models/llm/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
GEMINI_API_KEY = "PUT_YOUR_OWN"
MODEL_ID = "PUT_YOUR_OWN"

# --- HARDWARE SETTINGS ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Set to -1 to offload entirely to GPU (e.g., on Jetson Orin)
# Set to 0 to run entirely on CPU (e.g., standard Raspberry Pi)
LLM_GPU_LAYERS = 0  
LLM_THREADS = 4

# --- SYSTEM DIRECTORIES ---
SAVING_DIRECTORY = "./results_live"
deepseek_api = "PUT_YOUR_OWN"