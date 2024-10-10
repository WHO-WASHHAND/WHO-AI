import time
import cv2
from app.yolov8 import YOLOv8
import numpy as np
from app.yolov8.utils import draw_detections

list_steps = []

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
        ret, frame = cap.read()

        if not ret:
            print("Error: Cannot read frame.")
            break

        if frame_count % 25 == 0:
            # Update object localizer
            boxes, scores, class_ids = yolov8_detector(frame)

            if len(boxes) > 0:
                print(boxes, scores, class_ids)
                max_score_index = np.argmax(scores)
                max_box = boxes[max_score_index]

                max_score = scores[max_score_index]
                current_step = class_ids[max_score_index]
                list_steps.append(current_step)
                if tracker is None or (current_step != last_class_id):
                    bbox = tuple(int(value) for value in max_box)
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(frame, bbox)
                    last_class_id = current_step
                    last_scores = max_score

        if tracker is not None:
            start_time = time.time()
            success, bbox = tracker.update(frame)
            end_time = time.time()
            total_time = end_time - start_time
            print(f'Total processing time: {total_time:.2f} seconds')

            if success:
                bbox = np.array(bbox)
                print(bbox)
                combined_img = draw_detections(frame,[bbox],[last_scores],[last_class_id])
                _, buffer = cv2.imencode('.jpg', combined_img)
                frame_bytes = buffer.tobytes()
                socketio.emit('video_frame', frame_bytes)
                socketio.sleep(1 / fps)
                out.write(combined_img)
            else:
                # If tracking fails, reset tracker and last_step
                tracker = None
                last_class_id = None




        #combined_img = yolov8_detector.draw_detections(frame)
        #cv2.imshow("Detected Objects", combined_img) ####
        # Press key q to stop
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        # Increment frame counter
        frame_count += 1

    # Release resources and close windows
    cap.release()
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