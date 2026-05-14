import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────
#  get_points.py  —  Run this BEFORE tracker.py
# ─────────────────────────────────────────────────────────────────────

points = []
point_labels = [
    "P1: Left circle edge x halfway line",
    "P2: Right circle edge x halfway line",
    "P3: Left tip of penalty arc",
    "P4: Right tip of penalty arc",
]
point_colors = [
    (0,   215, 255),
    (0,   215, 255),
    (80,  140, 255),
    (80,  140, 255),
]

def click_event(event, x, y, flags, params):
    global frame # Ensure we modify the global frame variable
    if event == cv2.EVENT_LBUTTONDOWN:
        idx = len(points)
        if idx >= 4:
            print("Already have 4 points. Press 'q' to finish.")
            return

        points.append([x, y])
        print(f"  [{idx+1}/4] {point_labels[idx]}  ->  [{x}, {y}]")

        cv2.circle(frame, (x, y), 8, point_colors[idx], -1)
        cv2.circle(frame, (x, y), 8, (255, 255, 255), 2)
        cv2.putText(frame, str(idx + 1), (x + 12, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, point_colors[idx], 2)

        if len(points) >= 2:
            cv2.line(frame, tuple(points[-2]), tuple(points[-1]),
                     (255, 255, 255), 1, cv2.LINE_AA)
        if len(points) == 4:
            cv2.line(frame, tuple(points[3]), tuple(points[0]),
                     (255, 255, 255), 1, cv2.LINE_AA)
            print("\n  All 4 points selected! Press 'q' to finish.")
        else:
            cv2.putText(frame, f"Next -> {point_labels[len(points)]}",
                        (20, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow(win_name, frame)


video_path = r"C:\DL_PROJECT\data\demo_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"ERROR: Could not read video at: {video_path}")
    exit()

# =====================================================================
# --- NEW: STEP 1 - SCRUB THROUGH VIDEO TO FIND THE RIGHT FRAME ---
# =====================================================================
cv2.namedWindow("Select Frame")
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("\n" + "=" * 55)
print("  STEP 1: FIND THE RIGHT FRAME")
print("=" * 55)
print("  1. Use the slider at the top to scrub through the video.")
print("  2. Find a frame where all 4 points are clearly visible.")
print("  3. Press 'ENTER' to lock in this frame.")

selected_frame = None

def on_trackbar(val):
    global selected_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, val)
    ret, f = cap.read()
    if ret:
        selected_frame = f.copy()
        cv2.imshow("Select Frame", selected_frame)

# Create the slider
cv2.createTrackbar("Frame", "Select Frame", 0, total_frames - 1, on_trackbar)
on_trackbar(0) # Show the very first frame initially

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 13 or key == ord('c'): # 13 is the ENTER key
        break

cv2.destroyWindow("Select Frame")
cap.release()

if selected_frame is None:
    print("ERROR: No frame was captured.")
    exit()

# Pass our hand-picked frame to the clicking step
frame = selected_frame

# =====================================================================
# --- STEP 2: CLICK YOUR 4 POINTS ---
# =====================================================================
cv2.putText(frame, f"Click: {point_labels[0]}",
            (20, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

win_name = "Click 4 Points | Press Q when done"

print("\n" + "=" * 55)
print("  STEP 2: CALIBRATION  -  Click 4 landmark points")
print("=" * 55)
for i, lbl in enumerate(point_labels, 1):
    print(f"  {i}. {lbl}")
print("\n  Press 'q' after clicking all 4.\n")

cv2.imshow(win_name, frame)
cv2.setMouseCallback(win_name, click_event)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

print("\n" + "=" * 55)
if len(points) == 4:
    print("  Copy this into tracker.py as SOURCE_PTS:")
    print("=" * 55)
    print(f"\nSOURCE_PTS = np.array({points}, dtype=np.float32)\n")
else:
    print(f"  WARNING: Only {len(points)}/4 points. Re-run and click all 4.")
    print("=" * 55)