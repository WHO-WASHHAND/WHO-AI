import cv2
from ultralytics import YOLO

model = YOLO("models/yolov8_v2.1.pt")


def frame_to_image(frame):
    # Mã hóa frame thành định dạng JPEG
    success, encoded_image = cv2.imencode('.jpg', frame)
    if success:
        # Chuyển đổi ảnh đã mã hóa thành bytes
        return encoded_image.tobytes()
    else:
        print("Error: Cannot encode frame.")
        return None


def middle_frame_video(video_path, output_image_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame = total_frames // 2
    for frame in range(middle_frame - 5, middle_frame + 5):
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        ret, frame = cap.read()
        cap.release()
        if ret:
            cv2.imwrite(output_image_path, frame)
    cap.release()


object_custom_frequencies = {
    0: 7,
    1: 4,
    2: 4,
    3: 4,
    4: 4,
    5: 5
}


def generate_frames(cap, socketio):
    colors = [
        (255, 0, 0),  # Màu đỏ
        (0, 255, 0),  # Màu xanh lá
        (0, 0, 255),  # Màu xanh dương
        (255, 255, 0),  # Màu cyan
        (255, 0, 255),  # Màu magenta
        (0, 255, 255)  # Màu vàng
    ]
    # Khởi tạo CSRT tracker
    tracker = cv2.TrackerCSRT_create()
    tracking = False
    last_object_id = None  # ID của đối tượng trước đó
    frame_count = 0  # Biến đếm frame
    no_detection_count = 0  # Biến đếm số frame không phát hiện
    detected_objects = {}  # Lưu trữ các đối tượng và thời gian phát hiện
    object_frequencies = {}  # Lưu tần suất xuất hiện của các đối tượng
    object_color_map = {}
    executed_objects = set()
    object_detection_start_times = {}  # Thêm biến để lưu thời gian bắt đầu detect của đối tượng
    output_path = 'data/ouput_video_vp.webm'
    fourcc = cv2.VideoWriter_fourcc(*'vp80')  # Codec mp4
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    frame_duration = 1 / fps
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_video_time = frame_count * frame_duration  # Tính thời gian thực trong video

        if not tracking or frame_count % 25 == 0:
            results = model(frame)

            if results:
                detections = results[0]

                if detections.boxes is not None and len(detections.boxes.xyxy) > 0:
                    high_conf_indices = (detections.boxes.conf > 0.7).nonzero(as_tuple=True)[0]
                    high_conf_boxes = detections.boxes.xyxy[high_conf_indices]
                    high_conf_classes = detections.boxes.cls[high_conf_indices]

                    if high_conf_boxes.size(0) > 0:
                        no_detection_count = 0

                        for i in range(high_conf_boxes.size(0)):
                            current_bbox = high_conf_boxes[i].cpu().numpy()
                            current_bbox = (
                                int(current_bbox[0]), int(current_bbox[1]), int(current_bbox[2] - current_bbox[0]),
                                int(current_bbox[3] - current_bbox[1]))
                            current_object_id = int(high_conf_classes[i].item())

                            # Bỏ qua nếu đối tượng đã có trong danh sách thực hiện
                            if current_object_id in executed_objects:
                                continue

                            # Gán màu cho đối tượng nếu chưa có
                            if current_object_id not in object_color_map:
                                object_color_map[current_object_id] = colors[len(object_color_map) % len(colors)]

                            # Lưu thời gian phát hiện
                            if current_object_id not in detected_objects:
                                detected_objects[
                                    current_object_id] = current_video_time  # Lưu thời gian phát hiện theo thời gian video
                                object_detection_start_times[
                                    current_object_id] = current_video_time  # Lưu thời gian bắt đầu detect
                                object_frequencies[current_object_id] = 1  # Khởi tạo tần suất


                            else:
                                if current_video_time - detected_objects[current_object_id] <= \
                                        object_custom_frequencies[current_object_id]:
                                    object_frequencies[current_object_id] += 1
                                else:
                                    detected_objects[current_object_id] = current_video_time
                                    object_detection_start_times[
                                        current_object_id] = current_video_time  # Cập nhật thời gian bắt đầu detect
                                    object_frequencies[current_object_id] = 1
                                    socketio.emit('update_list_steps', {'steps': object_detection_start_times})
                            # Ghi lại các object có tần suất xuất hiện cao nhất trong 5 giây
                            for obj_id, freq in object_frequencies.items():
                                if freq > 2:
                                    executed_objects.add(obj_id)

                            # Khởi tạo tracker với bbox
                            if not tracking or current_object_id != last_object_id:
                                tracker.init(frame, current_bbox)
                                tracking = True
                                last_object_id = current_object_id
                                break
                    else:
                        no_detection_count += 1

                        if no_detection_count >= 2:
                            tracking = False
                            last_object_id = None


                else:
                    no_detection_count += 1

                    if no_detection_count >= 2:
                        tracking = False
                        last_object_id = None

        else:
            success, bbox = tracker.update(frame)
            if success:
                (x, y, w, h) = [int(v) for v in bbox]
                x = max(0, x)
                y = max(0, y)
                w = min(w, frame.shape[1] - x)
                h = min(h, frame.shape[0] - y)

                object_color = object_color_map.get(last_object_id, (255, 255, 255))

                cv2.rectangle(frame, (x, y), (x + w, y + h), object_color, 2)
                label = f'Step : {last_object_id + 1}'
                cv2.putText(frame, label, (x + 10, y - 10 if y - 10 > 10 else y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            object_color, 2)
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        socketio.emit('video_frame', frame_bytes)
        socketio.sleep(1 / fps)
        cv2.imshow('Object Tracking', frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    time_step = convert_dict_to_min_sec(object_detection_start_times)
    return time_step, output_path


def convert_dict_to_min_sec(input_dict):
    result = []
    for index, value in input_dict.items():
        seconds = value
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        result.append((index, f"{minutes}:{seconds:02d}"))
    return result


def increment_and_add_time(data):
    new_data = []

    # Tăng giá trị số nguyên trong từng tuple
    for index, (num, time_str) in enumerate(data):
        if index == 0:
            continue  # Bỏ qua phần tử đầu tiên
        new_num = num
        new_data.append((new_num, time_str))

    # Cộng thêm 5 giây vào thời gian của phần tử cuối
    last_num, last_time_str = new_data[-1]

    # Chuyển đổi thời gian từ định dạng '0:MM' thành giây
    minutes, seconds = map(int, last_time_str.split(':'))
    total_seconds = minutes * 60 + seconds + 5  # Cộng thêm 5 giây

    # Chuyển đổi lại thành định dạng '0:MM'
    new_minutes = total_seconds // 60
    new_seconds = total_seconds % 60

    # Cập nhật thời gian của phần tử cuối
    new_data[-1] = (last_num, f'{new_minutes}:{new_seconds:02d}')

    # Thêm phần tử mới với ID cuối cùng
    new_data.append((last_num + 1, f'{new_minutes}:{new_seconds:02d}'))

    return new_data
