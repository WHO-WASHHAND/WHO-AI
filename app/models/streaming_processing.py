import time
import cv2
from app.yolov8 import YOLOv8
import numpy as np
from app.yolov8.utils import draw_detections, xywh2xyxy, draw_box, draw_text, class_names
from app.yolov8.utils import colors


def process_video_with_yolov8(socketio,video_path, model_path,output_video_path, conf_thres=0.7, iou_thres=0.8):
    # Initialize the webcam
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    # Initialize YOLOv8 object detector
    yolov8_detector = YOLOv8(model_path, conf_thres=conf_thres, iou_thres=iou_thres)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    list_steps = []
    # Initialize VideoWriter to write the processed video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    # Variable
    frame_count = 0
    tracker = None
    last_class_id = None
    last_scores = None
    while True:
        # Read frame from the video

        '''for _ in range(2):
            cap.grab()'''
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame.")
            break
        if ret:
            frame_count += 1
            if frame_count % 25 == 0:
                # Update object localizer
                boxes, scores, class_ids = yolov8_detector(frame)

                if len(boxes) > 0:
                    print(boxes, scores, class_ids)
                    max_score_index = np.argmax(scores)
                    max_box = boxes[max_score_index]
                    max_score = scores[max_score_index]
                    current_step = class_ids[max_score_index]
                    list_steps.append([current_step,frame_count])

                    if tracker is None or (current_step != last_class_id):
                        bbox = tuple(int(value) for value in max_box)
                        tracker = cv2.TrackerCSRT_create()
                        tracker.init(frame, bbox)
                        last_class_id = current_step
                        last_scores = max_score

            if tracker is not None:
                success, bbox = tracker.update(frame)
                if success:
                    bbox = np.array(bbox)
                    x1, y1, x2, y2 = xywh2xyxy(bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), 2)
                    draw_box(frame, bbox, colors[last_class_id])
                    label = class_names[last_class_id]
                    caption = f'{label} '
                    img_height, img_width = frame.shape[:2]
                    font_size = min([img_height, img_width]) * 0.0006
                    text_thickness = int(min([img_height, img_width]) * 0.001)
                    draw_text(frame, caption, bbox, colors[last_class_id], font_size, text_thickness)
                else:
                    # If tracking fails, reset tracker and last_step
                    tracker = None
                    last_class_id = None

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            socketio.emit('video_frame', frame_bytes)
            socketio.sleep(1 / fps)
            out.write(frame)
            # Press key q to stop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

        # Increment frame counter

    print(list_steps)
    # Release resources and close windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()

'''if __name__ == "__main__":
    video_path = r"C:/Users/vtvan/OneDrive/Máy tính/video-rua-tay-1-10-20241002T075259Z-001/video-rua-tay-1-10/2.mp4"
    model_path = r"E:\HW\models\yolov8_v2.1.onnx"
    output_video_path = r"C:/Users/vtvan/OneDrive/Máy tính/output_video.mp4"
    start_time = time.time()
    process_video_with_yolov8(video_path, model_path,output_video_path)
    end_time = time.time()
    total_time = end_time - start_time
    print(f'Total processing time: {total_time:.2f} seconds')'''