# experiment_camera_spec.py

import cv2
import os
import json
from agent import run_agent  

OUTPUT_DIR = "experiment_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def resize_image(image, scale):
    try:
        if image is None:
            return None

        h, w = image.shape[:2]
        new_w = int(w * scale)
        new_h = int(h * scale)

        if new_w <= 0 or new_h <= 0:
            return None

        return cv2.resize(image, (new_w, new_h))
    except Exception:
        return None


def simulate_distance(image, scale):
    try:
        if image is None:
            return None

        h, w = image.shape[:2]

        small = resize_image(image, scale)
        if small is None:
            return None

        back = cv2.resize(small, (w, h))
        return back
    except Exception:
        return None


def safe_write_image(path, image):
    try:
        if image is None:
            return False
        return cv2.imwrite(path, image)
    except Exception:
        return False


def safe_run_agent(path):
    try:
        res = run_agent(path)
        if isinstance(res, dict):
            return res.get("result", res)
        return {"error": "invalid_agent_output"}
    except Exception as e:
        return {"error": str(e)}


def run_test(image_path):
    original = cv2.imread(image_path)
    if original is None:
        return

    scales = [1.0, 0.75, 0.5, 0.25, 0.1]

    results = []

    for scale in scales:
        print(f"\n=== TEST SCALE {scale} ===")

        try:
            resized = resize_image(original, scale)

            temp_path = f"{OUTPUT_DIR}/resized_{int(scale*100)}.jpg"
            saved = safe_write_image(temp_path, resized)

            if not saved:
                result_data = {"error": "image_write_failed"}
            else:
                result_data = safe_run_agent(temp_path)

        except Exception as e:
            result_data = {"error": str(e)}

        results.append({
            "type": "resolution",
            "scale": scale,
            "result": result_data
        })

        print("Result:", result_data)

    for scale in scales:
        print(f"\n=== DISTANCE SIMULATION {scale} ===")

        try:
            simulated = simulate_distance(original, scale)

            temp_path = f"{OUTPUT_DIR}/distance_{int(scale*100)}.jpg"
            saved = safe_write_image(temp_path, simulated)

            if not saved:
                result_data = {"error": "image_write_failed"}
            else:
                result_data = safe_run_agent(temp_path)

        except Exception as e:
            result_data = {"error": str(e)}

        results.append({
            "type": "distance",
            "scale": scale,
            "result": result_data
        })

        print("Result:", result_data)

    output_file = os.path.join(OUTPUT_DIR, "results.json")

    try:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
    except Exception:
        pass

    print("\nSaved results to:", output_file)


if __name__ == "__main__":
    run_test("pcbclear2.jpg")