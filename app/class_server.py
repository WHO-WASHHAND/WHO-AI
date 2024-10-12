import cv2
import socket
import struct
import time
import os


class VideoServer:
    def __init__(self, host='0.0.0.0', port=9999, video_source='http://117.2.164.10:28080/live/duong_test.flv',
                 fps=30, frame_size=(1080, 611), buffer_duration=5):
        self.host = host
        self.port = port
        self.video_source = video_source
        self.fps = fps
        self.frame_size = frame_size
        self.buffer_duration = buffer_duration
        self.server_socket = None
        self.client_socket = None
        self.client_addr = None

    # Hàm khởi tạo socket và chờ kết nối từ client
    def start_server(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print(f"Server started on {self.host}:{self.port}. Waiting for client connection...")

    # Hàm xử lý kết nối mới
    def accept_connection(self):
        try:
            self.client_socket, self.client_addr = self.server_socket.accept()
            print(f"Client connected: {self.client_addr}")
        except Exception as e:
            print(f"Error accepting connection: {e}")

    # Hàm lưu các khung hình thành tệp video ngắn
    def save_frames_to_video(self, frames, video_name):
        # Sử dụng mã hóa video (ví dụ: MP4)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_name, fourcc, self.fps, self.frame_size)

        for frame in frames:
            out.write(frame)  # Ghi từng khung hình vào tệp video

        out.release()  # Đóng file sau khi hoàn tất

    # Hàm gửi tệp video tới client
    def send_video_to_client(self, video_filename):
        with open(video_filename, 'rb') as f:
            video_data = f.read()
            # Đóng gói độ dài gói tin rồi gửi đi
            message = struct.pack("Q", len(video_data)) + video_data
            try:
                self.client_socket.sendall(message)  # Gửi tệp video tới client
            except Exception as e:
                print(f"Error sending video to client: {e}")
                self.client_socket.close()
                self.client_socket = None
                self.run()

    # Hàm xử lý luồng video
    def handle_video_stream(self):
        video_stream = cv2.VideoCapture(self.video_source)
        frame_buffer = []

        while video_stream.isOpened():
            start_time = time.time()

            # Thu thập khung hình trong khoảng buffer_duration (5 giây)
            while time.time() - start_time < self.buffer_duration:
                ret, frame = video_stream.read()
                if not ret:
                    break

                # Resize khung hình giữ nguyên tỉ lệ
                frame = cv2.resize(frame, self.frame_size)
                frame_buffer.append(frame)  # Thêm khung hình vào buffer

            # Khi đủ khung hình trong buffer_duration, lưu thành video ngắn và gửi đi
            if frame_buffer:
                video_filename = 'temp_video.avi'
                self.save_frames_to_video(frame_buffer, video_filename)
                frame_buffer = []  # Xóa buffer sau khi lưu

                # Gửi video cho client nếu có kết nối
                if self.client_socket:
                    self.send_video_to_client(video_filename)

                # Xóa tệp video sau khi gửi để tiết kiệm bộ nhớ
                os.remove(video_filename)

            if not ret:
                print("End of video stream")
                break

        video_stream.release()

    # Hàm chính để chạy server
    def run(self):
        self.start_server()

        while True:
            self.accept_connection()

            if self.client_socket:
                try:
                    self.handle_video_stream()
                except Exception as e:
                    print(f"Error during video streaming: {e}")
                finally:
                    print(f"Client {self.client_addr} disconnected.")
                    self.client_socket.close()
                    self.client_socket = None


# Chạy server
if __name__ == "__main__":
    server = VideoServer()
    server.run()
