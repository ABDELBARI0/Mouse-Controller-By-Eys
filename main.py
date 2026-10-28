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

