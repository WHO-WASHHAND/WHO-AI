import os

from flask import  request
from multiprocessing import Process
from app.class_server import VideoServer
from app.models.generate_frames import generate_frames, increment_and_add_time
from app.models.middle_frame_video import *
from app.models.streaming_processing import process_video_with_yolov8
from app.yolov8 import YOLOv8

iou_thres = 0.8
conf_thres = 0.7
model_path = r"models/yolov8_v2.1.onnx"
# Initialize YOLOv8 detector
yolov8_detector = YOLOv8(model_path, conf_thres=conf_thres, iou_thres=iou_thres)



def init_routes(app,socketio):
    @app.route('/detect', methods=['POST'])
    def detect():
        data = request.json
        video_url = data.get('url')
        if not video_url:
            return {'error': 'URL không hợp lệ'}, 400
        cap = cv2.VideoCapture(video_url)
        if not cap.isOpened():
            print("Error: Cannot open video.")
            exit(0)
        # Ouput Video & List Time Steps
        steps_in_seconds, output_path = generate_frames(cap,socketio)
        print(steps_in_seconds,output_path)
        # Middle Frame In Video
        output_image_path = 'data/ouput_video.jpg'
        middle_frame_video(video_url,output_image_path)
        steps_in_seconds = increment_and_add_time(steps_in_seconds)
        print(steps_in_seconds)
        # Call API
        #api_call_send_data_event((image_path,video_path, list_steps))
        return {'status': 'Started processing video '}, 200

    @app.route('/streaming', methods=['POST'])
    def streaming():
        data = request.json
        cam_url = data.get('url')
        cam_id = data.get('camera_id')
        if not cam_url:
            return {'error': 'IP không hợp lệ'}, 400


        output_video_path = os.path.join(os.path.dirname(__file__), r'../data/output_video_streaming.avi')

        # Tạo một Process mới để xử lý video
        # server = VideoServer(video_source=cam_url)
        # server.run()

        # p = Process(target=process_video_with_yolov8, args=(cam_id, socketio, cam_url, model_path, output_video_path))
        # p.start()  # Bắt đầu quá trình xử lý video không đồng bộ
        process_video_with_yolov8(
            cam_id=cam_id,
            socketio=socketio,
            output_video_path=output_video_path,
            video_path=cam_url,
            yolo_model=yolov8_detector,
        )
        # Trả về phản hồi ngay sau khi bắt đầu quá trình
        return {'status': 'Started processing video '}, 200

    @socketio.on('connect')
    def handle_connect():
        print('Client connected')

    @socketio.on('disconnect')
    def handle_disconnect():
        print('Client disconnected')
