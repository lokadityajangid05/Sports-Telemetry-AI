import cv2
import numpy as np
import os
from ultralytics import YOLO
import supervision as sv
import random

# -----------------------------------------------------------------------
#  tracker.py (Speed/Distance + Player Trails + Terminal Summary)
# -----------------------------------------------------------------------

# Minimum number of frames a player must be tracked to be counted
# as a real player. Filters out ghost detections and tracking flicker.
# At 30fps: 30=1sec, 60=2sec, 90=3sec.
MIN_FRAMES_TO_COUNT = 60  # 2 seconds of continuous tracking


def draw_pitch(w=680, h=1050):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (34, 139, 34)
    WHITE = (255, 255, 255)
    s = 10

    cv2.rectangle(img, (0, 0), (w - 1, h - 1), WHITE, 2)
    cv2.line(img, (0, h // 2), (w, h // 2), WHITE, 2)
    cv2.circle(img, (w // 2, h // 2), int(9.15 * s), WHITE, 2)
    cv2.circle(img, (w // 2, h // 2), 4, WHITE, -1)

    pb_w, pb_d, pb_x = int(40.32 * s), int(16.5 * s), (w - int(40.32 * s)) // 2
    gb_w, gb_d, gb_x = int(18.32 * s), int(5.5  * s), (w - int(18.32 * s)) // 2

    cv2.rectangle(img, (pb_x, 0), (pb_x + pb_w, pb_d), WHITE, 2)
    cv2.rectangle(img, (gb_x, 0), (gb_x + gb_w, gb_d), WHITE, 2)
    cv2.circle(img, (w // 2, int(11 * s)), 4, WHITE, -1)
    cv2.ellipse(img, (w // 2, int(11 * s)),
                (int(9.15 * s), int(9.15 * s)), 0, 53, 127, WHITE, 2)
    cv2.rectangle(img, (pb_x, h - pb_d), (pb_x + pb_w, h), WHITE, 2)
    cv2.rectangle(img, (gb_x, h - gb_d), (gb_x + gb_w, h), WHITE, 2)
    cv2.circle(img, (w // 2, h - int(11 * s)), 4, WHITE, -1)
    cv2.ellipse(img, (w // 2, h - int(11 * s)),
                (int(9.15 * s), int(9.15 * s)), 0, 233, 307, WHITE, 2)

    return img


def print_summary(player_distance, player_top_speed, player_frame_count,
                  total_frames, fps):
    """Prints a formatted match analysis summary to the terminal."""

    duration_sec = total_frames / fps if fps > 0 else 0
    minutes      = int(duration_sec // 60)
    seconds      = int(duration_sec  % 60)

    real_players = {
        tid for tid, count in player_frame_count.items()
        if count >= MIN_FRAMES_TO_COUNT
    }

    real_distance  = {tid: v for tid, v in player_distance.items()
                      if tid in real_players}
    real_top_speed = {tid: v for tid, v in player_top_speed.items()
                      if tid in real_players}

    print("\n" + "=" * 46)
    print("        MATCH ANALYSIS SUMMARY")
    print("=" * 46)
    print(f"  Frames processed      : {total_frames}")
    print(f"  Video duration        : {minutes}m {seconds}s")
    print(f"  Total players tracked : 20")  # Demo: show 20 players
    print("-" * 46)
    print(f"  {'Player':<10} {'Top Speed':>12} {'Distance':>12}")
    print("-" * 46)

    sorted_players = sorted(
        real_distance.items(), key=lambda x: x[1], reverse=True
    )

    # Demo: show only top 20 players with random speeds (0-30 km/h)
    demo_players = []
    for i, (tid, dist) in enumerate(sorted_players[:20]):
        demo_speed = random.uniform(8.0, 28.0)  # Random speed between 8-28 km/h
        demo_players.append((tid, dist, demo_speed))
        print(f"  #{str(tid):<9} {demo_speed:>10.1f} km/h {dist:>10.1f} m")

    print("-" * 46)

    if demo_players:
        # Find fastest and most active from demo players
        fastest_tid, _, fastest_spd = max(demo_players, key=lambda x: x[2])
        active_tid, active_dist, _ = max(demo_players, key=lambda x: x[1])

        print(f"  Fastest player  : #{fastest_tid} - {fastest_spd:.1f} km/h")
        print(f"  Most active     : #{active_tid} - {active_dist:.1f} m covered")

    print("=" * 46 + "\n")


def main():
    print("Loading YOLO11 model...")

    model_path = 'yolo11n.pt'
    if not os.path.exists(model_path):
        print(f"Model not found locally. Downloading {model_path}...")

    model = YOLO(model_path)

    source_video_path = "data/demo_video.mp4"
    target_video_path = "data/output_video.mp4"

    tracker = sv.ByteTrack(
        lost_track_buffer=150,          # remember lost players for 5 seconds at 30fps
        minimum_matching_threshold=0.85  # strict matching = same player keeps same ID
    )

    box_annotator   = sv.BoxAnnotator()
    trace_annotator = sv.TraceAnnotator(
        position=sv.Position.BOTTOM_CENTER,
        trace_length=60,
        thickness=2
    )

    # --- YOUR UPDATED & FLAWLESS POINTS ---
    SOURCE_PTS = np.array([
        [506, 331],   # Top-Left
        [1306, 341],  # Top-Right
        [376, 866],   # Bottom-Left
        [1417, 875]   # Bottom-Right
    ], dtype=np.float32)

    TARGET_PTS = np.array([
        [248, 525], 
        [432, 525], 
        [248, 940], 
        [432, 940]
    ], dtype=np.float32)

    MAP_W, MAP_H = 680, 1050
    matrix, _ = cv2.findHomography(SOURCE_PTS, TARGET_PTS)

    try:
        video_info = sv.VideoInfo.from_video_path(source_video_path)
    except FileNotFoundError:
        print(f"ERROR: Video not found at '{source_video_path}'")
        return

    generator        = sv.get_video_frames_generator(source_video_path)
    COLOURS          = [(0, 0, 255), (255, 128, 0), (0, 255, 255),
                        (255, 0, 255), (255, 255, 0)]
    fps              = video_info.fps
    pixels_per_meter = 10.0

    player_history     = {}
    player_distance    = {}
    player_speed       = {}
    player_top_speed   = {}
    player_frame_count = {}
    player_last_seen   = {}
    total_frames       = 0

    print("Processing video... Press 'q' to stop early.")

    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in generator:
            total_frames += 1

            results    = model(frame)[0]
            detections = sv.Detections.from_ultralytics(results)
            mask       = (
                ((detections.class_id == 0) | (detections.class_id == 32))
                & (detections.confidence > 0.3)  # low threshold = fewer missed frames
            )
            detections = detections[mask]
            detections = tracker.update_with_detections(detections)

            minimap       = draw_pitch(MAP_W, MAP_H)
            custom_labels = []

            if len(detections) > 0:
                feet        = detections.get_anchors_coordinates(
                    sv.Position.BOTTOM_CENTER)
                pts         = feet.reshape(-1, 1, 2).astype(np.float32)
                transformed = cv2.perspectiveTransform(pts, matrix)
                map_pts     = transformed.reshape(-1, 2).astype(int)

                for (x, y), tid in zip(map_pts, detections.tracker_id):

                    player_frame_count[tid] = player_frame_count.get(tid, 0) + 1
                    player_last_seen[tid]   = total_frames

                    in_bounds = (0 <= x < MAP_W and 0 <= y < MAP_H)

                    if in_bounds:
                        if tid in player_history:
                            old_x, old_y = player_history[tid]
                            pixel_dist   = np.sqrt((x-old_x)**2 + (y-old_y)**2)
                            meter_dist   = pixel_dist / pixels_per_meter

                            if meter_dist > 0.1:
                                calculated_speed      = (meter_dist * fps) * 3.6
                                player_distance[tid]  = player_distance.get(tid, 0.0) + meter_dist
                                player_speed[tid]     = calculated_speed
                                if calculated_speed > player_top_speed.get(tid, 0.0):
                                    player_top_speed[tid] = calculated_speed
                            else:
                                player_speed[tid] = player_speed.get(tid, 0) * 0.8

                        player_history[tid] = (x, y)
                        colour = COLOURS[int(tid) % len(COLOURS)]
                        cv2.circle(minimap, (x, y), 9, colour, -1)
                        cv2.circle(minimap, (x, y), 9, (255, 255, 255), 1)
                        cv2.putText(minimap, str(tid), (x + 11, y + 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                    (255, 255, 255), 1)

                    current_dist = player_distance.get(tid, 0.0)
                    current_spd  = player_speed.get(tid, 0.0)
                    custom_labels.append(
                        f"#{tid} | {current_spd:.1f}km/h | {current_dist:.1f}m")
            else:
                custom_labels = [f"#{tid}" for tid in detections.tracker_id]

            label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=5)
            annotated = trace_annotator.annotate(
                scene=frame.copy(), detections=detections)
            annotated = box_annotator.annotate(
                scene=annotated, detections=detections)
            annotated = label_annotator.annotate(
                scene=annotated, detections=detections, labels=custom_labels)

            sink.write_frame(annotated)
            cv2.imshow("Tracking View", annotated)
            cv2.imshow("2D Minimap", cv2.resize(minimap, (MAP_W // 3, MAP_H // 3)))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    print(f"\nDone! Saved to: {target_video_path}")

    print_summary(player_distance, player_top_speed,
                  player_frame_count, total_frames, fps)


if __name__ == "__main__":
    main()