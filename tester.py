import os
from PIL import Image

from vision_engine import VisionEngine
from reasoning_engine import ReasoningEngine
from parser_engine import ParserEngine
from judge_engine import JudgeEngine


TEST_IMAGES = [
    ("test_images/img1.jpg", "cut cake"),
    ("test_images/img2.jpg", "drink water"),
    ("test_images/img3.jpg", "store warm water"),
    ("test_images/img4.jpg", "read book"),
    ("test_images/img5.jpg", "write text"),
    ("test_images/img6.jpg", "play music"),
    ("test_images/img7.jpg", "clean floor"),
    ("test_images/img8.jpg", "cook food"),
    ("test_images/img9.jpg", "open parcel"),
    ("test_images/img10.jpg", "fix screw"),
]


def run():
    vision = VisionEngine()
    reasoner = ReasoningEngine()
    parser = ParserEngine()
    judge = JudgeEngine()

    score = 0

    for i, (path, task) in enumerate(TEST_IMAGES):
        print("\n" + "="*50)
        print(f"TEST {i+1}: {path}")
        print("TASK:", task)

        if not os.path.exists(path):
            print("❌ Missing image")
            continue

        image = Image.open(path).convert("RGB")

        # --- Vision ---
        scene = vision.analyze_scene(image)
        print("\n[Objects]")
        print(scene)

        # --- Reasoning ---
        reasoning = reasoner.get_affordance(task, scene)

        # --- Parse ---
        choice = parser.extract(reasoning)
        print("\n[Parsed Choice]", choice)

        # --- Judge ---
        verdict = judge.judge(task, scene, choice)
        print("\n[Judge]")
        print(verdict)

        if "CORRECT" in verdict.upper():
            score += 1

    print("\n" + "="*50)
    print(f"FINAL SCORE: {score}/{len(TEST_IMAGES)}")


if __name__ == "__main__":
    run()