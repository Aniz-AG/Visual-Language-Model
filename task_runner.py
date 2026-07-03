import csv
import os
import subprocess
import sys
from datetime import datetime

# ---------------- TASKS ----------------
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


# Add more runs here as you expand your test set.
# Each entry is (image_path, task_id).
TASK_RUNS = [
    ("test_images/1.png", 1),
    ("test_images/1.png", 2),
    ("test_images/1.png", 3),

    ("test_images/2.png", 4),
    ("test_images/2.png", 5),
    ("test_images/2.png", 6),

    ("test_images/3.png", 7),
    ("test_images/3.png", 8),
    ("test_images/3.png", 9),

    ("test_images/4.png", 10),
    ("test_images/4.png", 11),
    ("test_images/4.png", 12),

    ("test_images/5.jpeg", 13),
    ("test_images/5.jpeg", 14),
    ("test_images/5.jpeg", 15),

    ("test_images/6.png", 16),
    ("test_images/6.png", 17),
    ("test_images/6.png", 18),

    ("test_images/7.jpeg", 19),
    ("test_images/7.jpeg", 20),
    ("test_images/7.jpeg", 21),

    ("test_images/9.jpeg", 22),
    ("test_images/9.jpeg", 23),
    ("test_images/9.jpeg", 24),

    ("test_images/10.jpeg", 25),
    ("test_images/10.jpeg", 26),
    ("test_images/10.jpeg", 27),
]


def _write_run_log(log_file, *, image_path, image_name, task_id, task_name, command, return_code, output):
    separator = "=" * 90
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file.write(f"{separator}\n")
    log_file.write(f"Timestamp: {timestamp}\n")
    log_file.write(f"Image path: {os.path.abspath(image_path)}\n")
    log_file.write(f"Image: {image_name}\n")
    log_file.write(f"Task ID: {task_id}\n")
    log_file.write(f"Task: {task_name}\n")
    log_file.write(f"Command: {' '.join(command)}\n")
    log_file.write(f"Return code: {return_code}\n")
    log_file.write(f"{separator}\n")
    log_file.write(output.rstrip() + "\n\n")


def _get_python_executable():
    project_root = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(project_root, ".venv", "Scripts", "python.exe")
    if os.path.exists(venv_python):
        return venv_python
    return sys.executable


# ---------------- MAIN ----------------
def run(task_runs):
    if not task_runs:
        print("❌ No task runs found")
        return

    csv_file = "task_runner_raw_outputs.csv"
    log_file_name = "task_runner_full_run_log.txt"

    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "task_id", "task", "return_code", "raw_output"])

        with open(log_file_name, mode="w", encoding="utf-8", newline="\n") as log_file:
            log_file.write(f"Generated at: {datetime.now().isoformat(timespec='seconds')}\n")
            log_file.write(f"Run count: {len(task_runs)}\n\n")

            for image_path, task_id in task_runs:
                if not os.path.exists(image_path):
                    print(f"❌ Image not found: {image_path}")
                    continue

                if task_id not in taskid2name:
                    print(f"❌ Invalid task id: {task_id}")
                    continue

                image_name = os.path.basename(image_path)
                task_name = taskid2name[task_id]
                log_file.write(f"{'#' * 90}\n")
                log_file.write(f"RUN: {image_name} | task_id={task_id}\n")
                log_file.write(f"{'#' * 90}\n\n")

                print("\n" + "="*50)
                print(f"Running Task {task_id}: {task_name} | Image: {image_name}")

                python_executable = _get_python_executable()
                command = [
                    python_executable,
                    "main.py",
                    "--task-number",
                    str(task_id),
                    "--image-path",
                    image_path,
                ]

                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )

                output = (result.stdout or "") + ("\n" if result.stdout and result.stderr else "") + (result.stderr or "")

                print("\n[RAW OUTPUT]")
                print(output)

                _write_run_log(
                    log_file,
                    image_path=image_path,
                    image_name=image_name,
                    task_id=task_id,
                    task_name=task_name,
                    command=command,
                    return_code=result.returncode,
                    output=output,
                )

                writer.writerow([
                    image_name,
                    task_id,
                    task_name,
                    result.returncode,
                    output.replace("\n", " ")
                ])

    print("\n" + "="*50)
    print(f"✅ Raw outputs saved to {csv_file}")
    print(f"✅ Full transcript saved to {log_file_name}")


# ---------------- RUN ----------------
if __name__ == "__main__":
    run(TASK_RUNS)