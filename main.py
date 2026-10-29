# gaze_ui_combined.py
import cv2
import mediapipe as mp
import pyautogui as pya
import numpy as np
import math
import time

# ---------- CONFIG (UI) ----------
WIN_W = 1600
WIN_H = 900

TITLE_H = 60
STATUS_H = 50

PANEL_W = WIN_W // 2
PANEL_H = WIN_H - TITLE_H - STATUS_H

CAM_INDEX = 0  # camera device

# ---------- THEME ----------
def theme_C_palette():
    return {
        "bg": (255, 255, 255),
        "title_grad": ((245, 245, 247), (235, 245, 255)),
        "accent": (200, 220, 245),
        "border": (210, 210, 210),
        "text": (20, 20, 20)
    }

PALETTES = {
    3: ("Apple Gray + Blue Minimalist", theme_C_palette)
}

# ---------- HELPERS (UI) ----------
def draw_gradient_rect(img, x, y, w, h, c1, c2, vertical=True):
    if vertical:
        grad = np.linspace(0, 1, h).reshape(h, 1)
        grad = np.repeat(grad, w, axis=1)
    else:
        grad = np.linspace(0, 1, w).reshape(1, w)
        grad = np.repeat(grad, h, axis=0)

    c1 = np.array(c1, dtype=np.float32)
    c2 = np.array(c2, dtype=np.float32)
    img_part = (c1 * (1 - grad[..., None]) + c2 * grad[..., None]).astype(np.uint8)
    img[y:y+h, x:x+w] = img_part

def letterbox_resize(img, w, h):
    fh, fw = img.shape[:2]
    if fw == 0 or fh == 0:
        return np.full((h, w, 3), 255, dtype=np.uint8)
    scale = min(w/fw, h/fh)
    nw, nh = int(fw*scale), int(fh*scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    x0 = (w - nw) // 2
    y0 = (h - nh) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas

def draw_rounded_rect(img, x, y, w, h, color, thickness=2, r=12):
    cv2.rectangle(img, (x+r, y), (x+w-r, y+h), color, thickness)
    cv2.rectangle(img, (x, y+r), (x+w, y+h-r), color, thickness)
    cv2.circle(img, (x+r, y+r), r, color, thickness)
    cv2.circle(img, (x+w-r, y+r), r, color, thickness)
    cv2.circle(img, (x+r, y+h-r), r, color, thickness)
    cv2.circle(img, (x+w-r, y+h-r), r, color, thickness)
    
# ---------- UI BUILD ----------
def build_ui_canvas(palette, title_text, status_text, cam_frame, processed_frame, fps):
    ui = np.full((WIN_H, WIN_W, 3), palette["bg"], dtype=np.uint8)

    # Title Bar
    draw_gradient_rect(
        ui, 0, 0, WIN_W, TITLE_H,
        palette["title_grad"][0], palette["title_grad"][1],
        vertical=False
    )
    cv2.putText(ui, title_text, (24, TITLE_H//2 + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, palette["text"], 2, cv2.LINE_AA)

    # Left Panel (Camera)
    left_x, left_y = 20, TITLE_H + 10
    left_w, left_h = PANEL_W - 40, PANEL_H - 20

    cam_box = letterbox_resize(cam_frame, left_w, left_h)
    ui[left_y:left_y+left_h, left_x:left_x+left_w] = cam_box
    draw_rounded_rect(ui, left_x-4, left_y-4, left_w+8, left_h+8, palette["border"], 2, 14)
    cv2.putText(ui, "Camera View", (left_x+8, left_y+28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, palette["text"], 2, cv2.LINE_AA)

    # Right Panel
    right_x, right_y = PANEL_W + 20, TITLE_H + 10
    right_w, right_h = PANEL_W - 40, PANEL_H - 20

    proc_box = letterbox_resize(processed_frame, right_w, right_h)
    ui[right_y:right_y+right_h, right_x:right_x+right_w] = proc_box
    draw_rounded_rect(ui, right_x-4, right_y-4, right_w+8, right_h+8, palette["border"], 2, 14)
    cv2.putText(ui, "Mouse Tracking", (right_x+8, right_y+28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, palette["text"], 2, cv2.LINE_AA)

    # Status Bar
    status_y = WIN_H - STATUS_H
    cv2.rectangle(ui, (0, status_y), (WIN_W, WIN_H), palette["title_grad"][0], -1)
    cv2.putText(ui, status_text, (20, status_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, palette["text"], 2, cv2.LINE_AA)
    cv2.putText(ui, f"FPS: {fps:.1f}", (WIN_W - 160, status_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, palette["text"], 2, cv2.LINE_AA)

    # Accent strip
    cv2.rectangle(ui, (WIN_W-6, 0), (WIN_W, WIN_H), palette["accent"], -1)

    return ui

# ---------- Processed Panel (placeholder) ----------
def processed_panel(frame, palette):
    h, w = PANEL_H - 20, PANEL_W - 40
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (w, 50), palette["title_grad"][0], -1)
    cv2.putText(canvas, "", (18, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, palette["text"], 2, cv2.LINE_AA)
    return canvas

# ---------- Eye Tracking Globals ----------
mp_face_mesh = mp.solutions.face_mesh
# We'll open capture inside run function using CAM_INDEX
screen_w, screen_h = pya.size()

# smoothing
smooth_x, smooth_y = 0, 0
alpha_x = 0.2
alpha_y = 0.1
dead_zone = 30

# Calibration storage
calib_points = {}
calib_order = ["center", "left", "right", "top", "bottom"]
current_calib = 0
space_points = {}
space_points2 = {}

# blink detection
click_timer = 0
blink_threshold = 0.22
frames_to_click = 5
LEFT_EYE = [263, 387, 385, 362, 380, 373]

# preview window size
VIS_W, VIS_H = 500, 300
cursor_x, cursor_y = VIS_W // 2, VIS_H // 2

# --- helper functions for eye tracking ---
def get_rel_iris(landmarks):
    try:
        re_left, re_right = landmarks[33], landmarks[133]
        re_top, re_bot = landmarks[159], landmarks[145]
        iris1 = landmarks[468]

        eye_width = (re_right.x - re_left.x)
        eye_height = (re_bot.y - re_top.y)

        if abs(eye_width) < 1e-6 or abs(eye_height) < 1e-6:
            return None, None

        rel_x1 = (iris1.x - re_left.x) / eye_width
        rel_y1 = (iris1.y - re_top.y) / eye_height

        return rel_x1, rel_y1
    except Exception:
        return None, None

def eye_aspect_ratio(landmarks, eye_indices, frame=None, frame_w=None, frame_h=None):
    # compute EAR from 6 landmark indices (expects normalized landmarks with .x/.y)
    try:
        A = math.dist([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y],
                      [landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])
        B = math.dist([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y],
                      [landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y])
        C = math.dist([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y],
                      [landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])
        ear = (A + B) / (2.0 * C) if C != 0 else 0.0

        # optional debug drawing if frame provided
        if frame is not None and frame_w is not None and frame_h is not None:
            try:
                pts = [263, 387, 385, 362, 380, 373]
                for idx in pts:
                    cx, cy = int(landmarks[idx].x * frame_w), int(landmarks[idx].y * frame_h)
                    cv2.circle(frame, (cx, cy), 2, (0, 255, 255), -1)
            except Exception:
                pass

        return ear
    except Exception:
        return 0.0

# ---------- MAIN (integration) ----------
def run_ui_tester_with_gaze():
    global smooth_x, smooth_y, cursor_x, cursor_y
    global calib_points, current_calib, click_timer, space_points, space_points2

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Error: cannot open camera.")
        return

    palette_name, pal_func = PALETTES[3]
    palette = pal_func()

    prev_time = time.time()
    fps = 0.0
    title = "                                   Mouse Control via Eye Tracking - Theme 3"

    # Use mediapipe face mesh
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)

            # Build preview image for right panel
            vis = np.zeros((VIS_H, VIS_W, 3), dtype=np.uint8)
            cv2.putText(vis, "Cursor Preview", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            try:
                if results.multi_face_landmarks:
                    lm = results.multi_face_landmarks[0].landmark
                    rel_x, rel_y = get_rel_iris(lm)

                    # Blink detection
                    right_ear = eye_aspect_ratio(lm, LEFT_EYE, frame=frame, frame_w=frame_w, frame_h=frame_h)
                    if right_ear < blink_threshold:
                        click_timer += 1
                    else:
                        click_timer = 0

                    if click_timer > frames_to_click:
                        try:
                            pya.click()
                        except Exception:
                            pass
                        click_timer = 0
                        cv2.putText(frame, "BLINK CLICK!", (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    if rel_x is None or rel_y is None:
                        cv2.putText(frame, "Invalid iris detection", (30, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        # Calibration prompt
                        if current_calib < len(calib_order):
                            point_name = calib_order[current_calib]
                            cv2.putText(frame, f"Look at {point_name} and press SPACE", (30,30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                        # After calibrated
                        if len(calib_points) == 5:
                            left_x = min(calib_points["left"][0], calib_points["right"][0])
                            right_x = max(calib_points["left"][0], calib_points["right"][0])
                            top_y = min(calib_points["top"][1], calib_points["bottom"][1])
                            bottom_y = max(calib_points["top"][1], calib_points["bottom"][1])

                            rel_x = np.clip(rel_x, left_x, right_x)
                            rel_y = np.clip(rel_y, top_y, bottom_y)

                            # Map to screen coords
                            x_norm = np.interp(rel_x, [left_x, right_x], [0,screen_w])

                            dy = rel_y - calib_points["center"][1]
                            y_norm = np.interp(dy,
                                               [calib_points["top"][1] - calib_points["center"][1],
                                                calib_points["bottom"][1] - calib_points["center"][1]],
                                               [screen_h, 0])

                            x_norm = np.clip(x_norm, 0, screen_w - 1)
                            y_norm = np.clip(y_norm, 0, screen_h - 1)

                            if abs(smooth_x - x_norm) > dead_zone or abs(smooth_y - y_norm) > dead_zone:
                                smooth_x = smooth_x + alpha_x * (x_norm - smooth_x)
                                smooth_y = smooth_y + alpha_y * (y_norm - smooth_y)
                                try:
                                    pya.moveTo(int(smooth_x), int(smooth_y))
                                except Exception:
                                    pass

                                # Convert screen cursor pos → preview window pos
                                cursor_x = int((smooth_x / screen_w) * VIS_W)
                                cursor_y = int((smooth_y / screen_h) * VIS_H)

                            # Draw calibration rectangle on camera (debug)
                            try:
                                lx, rx = int(space_points["left"][0] * frame_w), int(space_points["right"][0] * frame_w)
                                ty, by = int(space_points["top"][1] * frame_h), int(space_points["bottom"][1] * frame_h)
                                cv2.rectangle(frame, (lx, ty), (rx, by), (0, 255, 0), 1)
                            except Exception:
                                pass

                    # Draw iris
                    try:
                        cx1, cy1 = int(lm[468].x * frame_w), int(lm[468].y * frame_h)
                        cv2.circle(frame, (cx1, cy1), 2, (0, 0, 255), -1)
                        cv2.circle(vis, (cursor_x, cursor_y), 8, (0, 255, 255), -1)
                    except Exception:
                        pass
                else:
                    cv2.putText(frame, "No face detected", (30, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            except Exception:
                pass

            # UI timing
            now = time.time()
            dt = now - prev_time
            prev_time = now
            fps = 0.9 * fps + 0.1 * (1.0/dt) if fps else 1.0/dt

            # Prepare panels for UI
            cam_panel = cv2.resize(frame, (PANEL_W - 40, PANEL_H - 20))
            proc_panel = letterbox_resize(vis, PANEL_W - 40, PANEL_H - 20)

            ui = build_ui_canvas(
                palette,
                title,
                f"Theme 3: {palette_name}",
                cam_panel,
                proc_panel,
                fps
            )

            cv2.imshow("UI Theme 3", ui)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == 32 and current_calib < len(calib_order):  # SPACE - calibrate
                if results and results.multi_face_landmarks:
                    lm = results.multi_face_landmarks[0].landmark
                    rel_x, rel_y = get_rel_iris(lm)
                    if rel_x is not None and rel_y is not None:
                        calib_points[calib_order[current_calib]] = (rel_x, rel_y)
                        try:
                            space_points[calib_order[current_calib]] = (lm[468].x, lm[468].y)
                            space_points2[calib_order[current_calib]] = (lm[473].x, lm[473].y)
                        except Exception:
                            pass
                        current_calib += 1
                    else:
                        print("⚠️ Tried to calibrate but iris not valid, skipped.")
            elif key == ord('s'):  # Recalibrate
                calib_points = {}
                space_points = {}
                space_points2 = {}
                current_calib = 0
                smooth_x = smooth_y = 0

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_ui_tester_with_gaze()