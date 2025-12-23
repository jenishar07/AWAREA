import os
import time
import math
import json
import cv2
from ultralytics import YOLO
from dotenv import load_dotenv

from google import genai
from google.genai import types

# ---------------- CONFIG ----------------
TARGET = "cell phone"        # must match a YOLO COCO class label for yolov8n.pt
SCAN_SECONDS = 4.0
CONF_MIN = 0.35

CONTEXT_MAX = 12
NEARBY_MAX_DIST = 0.30

# Use the exact model id you listed
GEMINI_MODEL = "models/gemini-2.5-flash"
# --------------------------------------


def center_xyxy(x1, y1, x2, y2):
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def norm_dist(ax, ay, bx, by, w, h):
    dx = (ax - bx) / max(1.0, w)
    dy = (ay - by) / max(1.0, h)
    return math.sqrt(dx * dx + dy * dy)


def region_bucket(cx, cy, w, h):
    """Returns: top-left, top, top-right, left, center, right, bottom-left, bottom, bottom-right"""
    rx = cx / max(1.0, w)
    ry = cy / max(1.0, h)

    horiz = "left" if rx < 0.33 else "right" if rx > 0.66 else "center"
    vert = "top" if ry < 0.33 else "bottom" if ry > 0.66 else "center"

    if horiz == "center" and vert == "center":
        return "center"
    if horiz == "center":
        return vert
    if vert == "center":
        return horiz
    return f"{vert}-{horiz}"


def collect_best_scan(yolo, cap):
    """
    Scan camera for SCAN_SECONDS.
    Returns:
      best_target: dict or None
      best_context: list[dict] from the SAME frame where best_target was best
    """
    start = time.time()
    best_target = None
    best_context = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        res = yolo(frame, verbose=False)[0]

        frame_best = None
        frame_context = []

        for box in res.boxes:
            cls_id = int(box.cls[0].item())
            label = yolo.names[cls_id]
            conf = float(box.conf[0].item())

            if conf < CONF_MIN:
                continue

            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            cx, cy = center_xyxy(x1, y1, x2, y2)

            item = {
                "label": label,
                "confidence": round(conf, 3),
                "center": [round(cx, 1), round(cy, 1)],
                "region": region_bucket(cx, cy, w, h),
                "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            }

            if label.lower() == TARGET.lower():
                if frame_best is None or conf > frame_best["confidence"]:
                    frame_best = item
            else:
                frame_context.append(item)

        # If this frame has the target, maybe update the global best target
        if frame_best:
            if best_target is None or frame_best["confidence"] > best_target["confidence"]:
                best_target = frame_best
                best_target["frame_size"] = [w, h]

                # Keep nearest context objects to target center (within distance threshold)
                tcx, tcy = best_target["center"]
                near = []
                for it in frame_context:
                    d = norm_dist(tcx, tcy, it["center"][0], it["center"][1], w, h)
                    if d <= NEARBY_MAX_DIST:
                        near.append((d, it))

                near.sort(key=lambda x: x[0])
                best_context = [it for _, it in near][:CONTEXT_MAX]

        # Optional live preview during scan
        elapsed = time.time() - start
        cv2.putText(
            frame,
            f"SCANNING... {min(elapsed, SCAN_SECONDS):.1f}/{SCAN_SECONDS:.1f}s | Target: {TARGET}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Scan Mode", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if elapsed >= SCAN_SECONDS:
            break

    return best_target, best_context


def gemini_describe(target, context):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GOOGLE_API_KEY in environment/.env")

    client = genai.Client(api_key=api_key)

    payload = {
        "target_query": TARGET,
        "best_target_detection": target,
        "nearby_objects": context,
        "notes": [
            "Coordinates are in pixels of the camera frame.",
            "region is a rough camera-relative bucket like top-left/center/bottom-right.",
        ],
    }

    # STRONG prompt to prevent fragment answers like "The cell phone"
    prompt = f"""
You are a vision assistant.

TASK:
Describe where the target object is located in the scene using the detection data.

STRICT RULES:
- You MUST write one complete English sentence.
- You MUST mention the target name: "{TARGET}".
- You MUST mention its camera-relative location using the region (top-left/center/bottom-right/etc).
- If any nearby_objects exist, mention at least ONE of them as context (e.g., "near a laptop").
- Do NOT reply with only the object name.
- Do NOT output a list. Output exactly ONE sentence.

DATA:
{json.dumps(payload, indent=2)}
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=80,
        ),
    )

    text = (response.text or "").strip()

    # Hard safety fallback if response is still too short
    if len(text.split()) < 6:
        region = target.get("region", "somewhere in the view")
        text = f"The {TARGET} appears around the {region} of the view."

    return text


def main():
    load_dotenv()

    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ GOOGLE_API_KEY not found. Create .env with GOOGLE_API_KEY=YOUR_KEY")
        return

    yolo = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    best_target, best_context = collect_best_scan(yolo, cap)

    cap.release()
    cv2.destroyAllWindows()

    if not best_target:
        print(f"❌ Could not find '{TARGET}' in the scanned area.")
        return

    description = gemini_describe(best_target, best_context)

    print("\n✅ AI Description:")
    print(description)


if __name__ == "__main__":
    main()
