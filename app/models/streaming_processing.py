import time
import cv2
import numpy as np
from app.yolov8 import YOLOv8
from app.yolov8.utils import draw_box, draw_text, class_names, xywh2xyxy, colors


def initialize_video_writer(video_path, output_video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return None, None, None, None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    return cap, out, width, height


def initialize_tracker():
    return cv2.TrackerCSRT_create()


def reset_tracking_vars():
    return None, None, 0, []


def update_tracker(yolov8_detector, frame, tracker, last_class_id, last_scores, list_steps, frame_count):
    boxes, scores, class_ids = yolov8_detector(frame)
    if len(boxes) > 0:
        max_score_index = np.argmax(scores)
        max_box = boxes[max_score_index]
        max_score = scores[max_score_index]
        current_step = class_ids[max_score_index]
        list_steps.append([current_step, frame_count])
        if tracker is None or current_step != last_class_id:
            bbox = tuple(int(value) for value in max_box)
            tracker = initialize_tracker()
            tracker.init(frame, bbox)
            return tracker, current_step, max_score
    return tracker, last_class_id, last_scores


def process_tracking(frame, tracker, last_class_id):
    success, bbox = tracker.update(frame)
    if success:
        bbox = np.array(bbox)
        x1, y1, x2, y2 = xywh2xyxy(bbox)
        draw_box(frame, bbox, colors[last_class_id])
        label = class_names[last_class_id]
        caption = f'{label} '
        img_height, img_width = frame.shape[:2]
        font_size = min([img_height, img_width]) * 0.0006
        text_thickness = int(min([img_height, img_width]) * 0.001)
        draw_text(frame, caption, bbox, colors[last_class_id], font_size, text_thickness)
        return True
    return False


def process_video_with_yolov8(socketio, video_path, model_path, output_video_path, conf_thres=0.7, iou_thres=0.8):
    # Initialize video capture and writer
    cap, out, width, height = initialize_video_writer(video_path, output_video_path)
    if cap is None:
        return

    # Initialize YOLOv8 detector
    yolov8_detector = YOLOv8(model_path, conf_thres=conf_thres, iou_thres=iou_thres)

    # Initialize variables
    frame_count, check_status_reset = 0, 0
    tracker, last_class_id, last_scores, list_steps = reset_tracking_vars()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame.")
            break

        frame_count += 1

        # Process every 25th frame for object detection
        if frame_count % 25 == 0:
            tracker, last_class_id, last_scores = update_tracker(
                yolov8_detector, frame, tracker, last_class_id, last_scores, list_steps, frame_count
            )
            if tracker is None:
                check_status_reset += 1
            else:
                check_status_reset = 0
            if check_status_reset ==2:
                tracker = None
            if check_status_reset >= 5:
                print("Resetting due to no objects detected.")
                tracker, last_class_id, last_scores, list_steps = reset_tracking_vars()
                frame_count, check_status_reset = 0, 0

        # Continue tracking
        if tracker is not None:
            if not process_tracking(frame, tracker, last_class_id):
                tracker, last_class_id = None, None

        # Emit frame to the socket
        _, buffer = cv2.imencode('.jpg', frame)
        socketio.emit('video_frame', buffer.tobytes())
        socketio.sleep(1 / cap.get(cv2.CAP_PROP_FPS))

        # Write frame to the output video
        out.write(frame)

        # Stop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Final Steps:", list_steps)
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

