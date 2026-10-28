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

