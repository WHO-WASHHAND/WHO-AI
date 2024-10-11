import os
import time
import cv2
import numpy as np
from collections import Counter
from app.models.generate_frames import middle_frame_video
from app.yolov8 import YOLOv8
from app.yolov8.utils import draw_box, draw_text, class_names, xywh2xyxy, colors, frame_to_time


def initialize_video_writer(video_path, output_video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return None, None, None, None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'vp80')  # Codec VP8 cho WebM
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    return cap, out, width, height


def initialize_tracker():
    return cv2.TrackerCSRT_create()


def reset_tracking_vars():
    return None, None, 0, [], []


def update_tracker(socketio,yolov8_detector, frame, tracker, last_class_id, last_scores, list_steps, frame_count,fps,check_status_reset,stream_step):
    boxes, scores, class_ids = yolov8_detector(frame)
    if len(boxes) > 0:
        check_status_reset = 0
        max_score_index = np.argmax(scores)
        max_box = boxes[max_score_index]
        max_score = scores[max_score_index]
        current_step = class_ids[max_score_index]
        list_steps.append([current_step, frame_count])
        stream_step = process_steps_with_threshold(socketio,list_steps,fps)# Lưu bước hiện tại
        if  current_step != last_class_id:
            bbox = tuple(int(value) for value in max_box)
            tracker = initialize_tracker()
            tracker.init(frame, bbox)
            return tracker, current_step, max_score,check_status_reset,stream_step
    else:
        check_status_reset += 1
        tracker = None
    return tracker, last_class_id, last_scores , check_status_reset,stream_step


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


def process_steps_with_threshold(socketio,list_steps, fps, time_threshold=150):
    completed_steps = []
    step_window = []
    seen_steps = set()  # Set để theo dõi các step đã được thực hiện
    start_time = None

    for i, (step, time) in enumerate(list_steps):
        if start_time is None:
            start_time = time  # Khởi tạo thời gian bắt đầu cho window

        # Thêm step vào cửa sổ theo dõi
        step_window.append((step, time))

        # Kiểm tra nếu đã đạt ngưỡng thời gian hoặc nếu đó là phần tử cuối cùng
        if time - start_time >= time_threshold or i == len(list_steps) - 1:
            # Tính tần suất xuất hiện của các step trong cửa sổ
            step_counts = Counter([s for s, _ in step_window])
            most_common_step, count = step_counts.most_common(1)[0]

            # Chỉ ghi lại nếu step chưa được thực hiện trước đó
            if most_common_step not in seen_steps:
                step_start_time = step_window[0][1]
                step_end_time = step_window[-1][1]

                completed_steps.append((most_common_step, frame_to_time(step_end_time, fps)))
                seen_steps.add(most_common_step)  # Đánh dấu step là đã được thực hiện

            # Reset lại cửa sổ theo dõi cho các step tiếp theo
            step_window = []
            start_time = None
    completed_steps = [(int(step[0]), step[1]) for step in completed_steps]
    socketio.emit('list_step_streaming', {'steps': completed_steps})
    return completed_steps


def process_video_with_yolov8(socketio, video_path, model_path, output_video_path, conf_thres=0.7, iou_thres=0.8):
    # Initialize video capture and writer
    cap, out, width, height = initialize_video_writer(video_path, output_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if cap is None:
        return

    # Initialize YOLOv8 detector
    yolov8_detector = YOLOv8(model_path, conf_thres=conf_thres, iou_thres=iou_thres)

    # Initialize variables
    frame_count, check_status_reset = 0, 0
    tracker, last_class_id, last_scores, list_steps,stream_step = reset_tracking_vars()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame.")
            break

        frame_count += 1

        # Process every 25th frame for object detection
        if frame_count % fps == 0:
            tracker, last_class_id, last_scores, check_status_reset,stream_step = update_tracker(socketio,
                yolov8_detector, frame, tracker, last_class_id, last_scores, list_steps, frame_count,fps,check_status_reset,stream_step
            )
            if check_status_reset >= 3 and len(list_steps) > 0:
                output_steps = stream_step
                # Setup data for API
                output_video_path = os.path.join(os.path.dirname(__file__), r'../../data/output_video_streaming.webm')
                output_img_path = os.path.join(os.path.dirname(__file__), r'../../data/output_video_streaming.jpg')
                middle_frame_video(output_video_path, output_img_path)
                # call API để gửi completed_steps nếu cần
                print(output_img_path,output_video_path,output_steps)
                # Reset biến
                print("Resetting due to no objects detected.")
                tracker, last_class_id, last_scores, list_steps,stream_step = reset_tracking_vars()
                frame_count, check_status_reset = 0, 0

        # Continue tracking
        if tracker is not None:
            process_tracking(frame, tracker, last_class_id)
        # Emit frame to the socket
        _, buffer = cv2.imencode('.jpg', frame)
        socketio.emit('streaming_data', buffer.tobytes())
        socketio.sleep(1 / cap.get(cv2.CAP_PROP_FPS))
        # Write frame to the output video
        out.write(frame)
        # Stop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
