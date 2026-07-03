# Visual-Language-Model
The problem is: Given an image and a user query, select the most suitable object in the image that satisfies the query. This is an affordance detection task.

Constraints:
- Must run on edge devices (e.g., Raspberry Pi)
- Strict memory and compute limitations
- Offline pipeline (no API calls, no large LLMs)

Pipeline evolution and experiments:

1. YOLO + Small Language Model (SLM)
   - YOLO used for object detection
   - SLM (Phi-2, LLaMA) used for reasoning over detected objects
   - Limitation: YOLO cannot detect novel/unseen objects → poor generalization

2. Vision-Language Model (VLM) approach
   - Used Florence-2 base model
   - Accepts both image and text input
   - Uses pixel embeddings + text embeddings
   - Performs joint reasoning

   Why Florence-2:
   - Lightweight compared to other VLMs
   - Strong grounding + captioning ability
   - Better suited for edge compared to large VLMs

   Limitations:
   - Struggles with multi-object scenes
   - Dense region captioning sometimes misses objects

3. Hybrid Pipeline (Florence + SLM)
   - Florence used only for perception (object detection + attributes)
   - SLM used for reasoning

4. Attribute-based (Affordance-based) reasoning
   - Instead of object names → use attributes
   - Example:
     Query: "What can store warm water?"
     Objects: plastic bottle vs metallic bottle
     Decision depends on material properties

   - Florence generates object + attributes
   - Attributes passed to SLM for reasoning

5. Two-pass Florence pipeline
   - Pass 1: Object detection + segmentation
   - Pass 2: Dense region captioning on segmented objects
   - Improves multi-object understanding

6. SLM details:
   - Models used:
     - Phi-2 (earlier)
     - LLaMA (later)
   - Reason for shift:
     - Phi-2 hallucinated and over-reasoned
     - LLaMA produced more stable outputs

   - Memory constraints:
     - 1B parameter models

7. NLP-assisted pipeline:
   - Used spaCy for:
     - Attribute extraction from Florence captions
     - Query parsing
   - Reduces reasoning burden on SLM

8. Chain-of-Thought reasoning (3 steps):
   - Step 1: Extract required features from query
   - Step 2: Match object attributes with required features
   - Step 3: Rank importance and select best object

9. Evaluation:
   - Dataset: 50+ self-captured household images
   - Each image tested with 3 queries

10. Challenges:
   - Edge constraints (RAM, compute)
   - Model size vs performance tradeoff
   - Multi-object reasoning difficulty
   - Attribute extraction noise
   - Hallucination in SLMs
   - Segmentation inaccuracies
