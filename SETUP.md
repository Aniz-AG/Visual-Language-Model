# Setup Guide (Mac M1)

## Files Needed

- `florence_live_detection.py`
- `COT.py`
- `test.jpg` (any test image)

## Setup Commands

```bash
# 1. Create environment
conda create -n affordance python=3.10 -y
conda activate affordance

# 2. Install PyTorch
pip install torch torchvision

# 3. Install llama-cpp with Metal
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --no-cache-dir

# 4. Install other packages
pip install transformers pillow huggingface_hub einops timm

# 5. Run (models download automatically on first run)
python florence_live_detection.py --task-number 15 --image-path test.jpg
```

## Notes

- First run downloads ~2.5GB of models (Florence-2 + Phi-2)
- Use `--task-number` 1-15 for different tasks
- Pass any image with `--image-path`
