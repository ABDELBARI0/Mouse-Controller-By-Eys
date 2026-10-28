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