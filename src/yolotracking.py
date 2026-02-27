import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

# ---- Choose which object classes you care about ----
# Change this list for your project, e.g. ["bottle"], ["person"], ["cup", "bottle"], etc.
ALLOWED_CLASSES = ["bottle", "cup"]

# ----- MediaPipe setup -----
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ----- YOLO model -----
yolo_model = YOLO("yolov8n.pt")  # change model if you want (e.g. yolov8s.pt)


def distance_point_to_line(pt, line_p, line_d):
    """Shortest distance from pt to infinite line defined by point line_p and direction line_d."""
    v = pt - line_p
    cross = abs(v[0] * line_d[1] - v[1] * line_d[0])
    denom = np.linalg.norm(line_d) + 1e-6
    return cross / denom


def projection_scalar(pt, line_p, line_d):
    """Scalar t for projection of pt on line: line_p + t * line_d."""
    v = pt - line_p
    denom = np.dot(line_d, line_d) + 1e-6
    return np.dot(v, line_d) / denom


def detect_objects_yolo(frame, conf_threshold=0.5, allowed_classes=None):
    """
    Run YOLO on the frame.
    Returns a list of dicts:
      {
        "bbox": (x1, y1, x2, y2),
        "center": (cx, cy),
        "label": class_name,
        "conf": confidence
      }
    Only keeps objects whose label is in allowed_classes (if provided).
    """
    results = yolo_model(frame, verbose=False)[0]  # first result

    objects = []
    if results.boxes is None:
        return objects

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())
        if conf < conf_threshold:
            continue

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        cls_id = int(box.cls[0].cpu().numpy())
        label = yolo_model.names.get(cls_id, str(cls_id))

        # ----- Filter by class name -----
        if allowed_classes is not None and label not in allowed_classes:
            continue

        objects.append({
            "bbox": (x1, y1, x2, y2),
            "center": (cx, cy),
            "label": label,
            "conf": conf
        })

    return objects


def main():
    cap = cv2.VideoCapture(4)  # change index if needed

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # ---------- Hand detection ----------
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            index_tip = None
            base_point = None

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    tip = hand_landmarks.landmark[8]  # index fingertip
                    mcp = hand_landmarks.landmark[5]  # index MCP (base)

                    index_tip = np.array([int(tip.x * w), int(tip.y * h)])
                    base_point = np.array([int(mcp.x * w), int(mcp.y * h)])

                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

            # ---------- YOLO object detection (only ALLOWED_CLASSES) ----------
            objects = detect_objects_yolo(
                frame,
                conf_threshold=0.5,
                allowed_classes=ALLOWED_CLASSES
            )

            # Draw all detected (filtered) objects
            for obj in objects:
                x1, y1, x2, y2 = obj["bbox"]
                cx, cy = obj["center"]
                label = obj["label"]
                conf = obj["conf"]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                cv2.putText(
                    frame,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

            pointed_obj = None  # will store object + z

            # ---------- Pointing line & choose object ----------
            if index_tip is not None and base_point is not None:
                line_p = base_point.astype(float)
                line_d = (index_tip - base_point).astype(float)

                # Draw long pointing line
                p2 = (index_tip + 5 * (index_tip - base_point)).astype(int)
                cv2.line(
                    frame,
                    tuple(base_point.astype(int)),
                    tuple(p2),
                    (255, 0, 0),
                    2
                )
                cv2.circle(frame, tuple(index_tip.astype(int)), 6, (255, 0, 0), -1)

                if len(objects) > 0:
                    min_dist = 999999.0
                    best_obj = None
                    best_t = None

                    for obj in objects:
                        cx, cy = obj["center"]
                        center = np.array([cx, cy], dtype=float)

                        t = projection_scalar(center, line_p, line_d)
                        if t <= 0:
                            continue  # object is behind the hand along this ray

                        dist = distance_point_to_line(center, line_p, line_d)
                        if dist < min_dist and dist < 30:  # threshold in pixels
                            min_dist = dist
                            best_obj = obj
                            best_t = t

                    if best_obj is not None:
                        x1, y1, x2, y2 = best_obj["bbox"]
                        cx, cy = best_obj["center"]
                        label = best_obj["label"]
                        conf = best_obj["conf"]

                        pointed_obj = {
                            "bbox": (x1, y1, x2, y2),
                            "center": (cx, cy),
                            "label": label,
                            "conf": conf,
                            "z": float(best_t)  # distance along pointing line
                        }

            # ---------- Highlight pointed object ----------
            if pointed_obj is not None:
                x1, y1, x2, y2 = pointed_obj["bbox"]
                cx, cy = pointed_obj["center"]
                label = pointed_obj["label"]
                conf = pointed_obj["conf"]
                z = pointed_obj["z"]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(
                    frame,
                    f"{label}: x={cx}, y={cy}, z={z:.1f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

            cv2.imshow("Hand Pointing with YOLO (filtered classes)", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
