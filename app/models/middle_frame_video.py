# app/models/middle_frame_video.py
import cv2


# Hàm xử lý frame giữa video
def middle_frame_video(video_path, output_image_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame = total_frames // 2

    for frame in range(middle_frame - 15, middle_frame + 15):
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        ret, frame = cap.read()
        cap.release()
        if ret:
            cv2.imwrite(output_image_path, frame)
