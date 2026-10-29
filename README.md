# Mouse Controller by Eye Tracking

Control the mouse cursor with your eyes using a webcam. The application detects facial and iris landmarks with MediaPipe, calibrates your gaze to your screen, and uses a sustained blink to click.

## Features

- Tracks iris position through your webcam.
- Calibrates gaze against five points: center, left, right, top, and bottom.
- Smooths cursor movement and displays a live camera view and cursor preview.
- Performs a left click when a blink is detected.
- Supports recalibration while the application is running.

## Requirements

- Python with versions supported by the pinned packages in `requirements.txt`.
- A working webcam.
- A desktop session where the application can access the webcam and control the mouse.

The project uses OpenCV, MediaPipe, NumPy, and PyAutoGUI. `requirements.txt` contains the pinned dependencies.

## Setup

1. Clone or download this repository.
2. From the project directory, create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   Windows PowerShell:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

   macOS or Linux:

   ```bash
   source .venv/bin/activate
   ```

3. Install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

## Run

Start the application from the project directory:

```bash
python main.py
```

Allow webcam access if prompted. The default camera device is index `0`; change `CAM_INDEX` near the top of `main.py` if you need to use a different camera.

## Calibration and controls

1. Look at the point requested in the camera window.
2. Press **Space** to record that point. Repeat for **center**, **left**, **right**, **top**, and **bottom**.
3. Once all five points have been recorded, move your gaze to control the cursor.
4. Blink and hold briefly to trigger a left click.
5. Press **S** to clear the calibration and start it again.
6. Press **Esc** to exit.

For best results, keep your face visible and reasonably still during calibration, and make sure the camera has a clear view of your eyes.

## Notes

- The application displays the webcam feed and a cursor preview in its window.
- The cursor can move and blink clicks can interact with the active desktop. Close the application when you need to stop mouse input.
- Camera frames are processed locally by the application; the project does not include a network upload feature.

## Troubleshooting

- **Camera does not open:** Check that it is connected, not in use by another application, and that the selected `CAM_INDEX` is correct.
- **Face or iris is not detected:** Improve lighting, face the camera, and keep your eyes unobstructed.
- **Cursor movement is inaccurate:** Press **S** and redo all five calibration points while looking directly at each requested point.
- **Mouse control is blocked:** Check your operating system's camera and accessibility/input permissions.
