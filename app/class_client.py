import cv2
import threading
import socket
import queue
import struct
import time
import os


class VideoClient:
    def __init__(self, host='localhost', port=9999):
        self.host = host
        self.port = port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.video_queue = queue.Queue(maxsize=10)

    def connect(self):
        try:
            self.client_socket.connect((self.host, self.port))
            print(f"Connected to server at {self.host}:{self.port}")
        except Exception as e:
            print(f"Connection error: {e}")
            self.client_socket.close()

    def read_video(self):
        data = b""
        payload_size = struct.calcsize("Q")

        while True:
            # Nhận dữ liệu cho tới khi đủ payload_size (độ dài kích thước gói tin)
            while len(data) < payload_size:
                packet = self.client_socket.recv(64 * 1024)  # Nhận dữ liệu từ server
                if not packet:
                    break
                data += packet

            # Giải nén độ dài gói tin
            if len(data) < payload_size:
                continue
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("Q", packed_msg_size)[0]

            # Nhận toàn bộ dữ liệu video
            while len(data) < msg_size:
                data += self.client_socket.recv(64 * 1024)

            # Giải mã dữ liệu video
            video_data = data[:msg_size]
            data = data[msg_size:]

            # Lưu dữ liệu video vào tệp tạm thời
            video_filename = f'temp_received_video_{time.time()}.avi'
            with open(video_filename, 'wb') as f:
                f.write(video_data)

            # Đưa tệp video vào hàng đợi
            if not self.video_queue.full():
                self.video_queue.put(video_filename)

    def show_video(self):
        while True:
            if not self.video_queue.empty():
                # Lấy tệp video từ hàng đợi
                video_filename = self.video_queue.get()

                # Mở và phát video trực tiếp từ tệp video
                cap = cv2.VideoCapture(video_filename)

                if not cap.isOpened():
                    print(f"Không thể mở video: {video_filename}")
                    continue

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Hiển thị trực tiếp video (phải ở luồng chính)
                    cv2.imshow('Video', frame)

                    # Kiểm tra phím nhấn 'q' để thoát
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        self.client_socket.close()
                        cap.release()
                        cv2.destroyAllWindows()
                        return  # Thoát hàm show_video()

                # Giải phóng tài nguyên video
                cap.release()

                # Xóa tệp video sau khi phát
                os.remove(video_filename)

        # Đóng socket và cửa sổ OpenCV khi kết thúc
        self.client_socket.close()
        cv2.destroyAllWindows()

    def run(self):
        self.connect()

        # Tạo thread để đọc video (chỉ đọc dữ liệu, không hiển thị)
        t1 = threading.Thread(target=self.read_video, daemon=True)

        # Khởi động thread đọc video
        t1.start()

        # Hiển thị video trong thread chính (phải thực thi trong luồng chính)
        self.show_video()

        # Chờ thread đọc video kết thúc
        t1.join()


# Chạy client
if __name__ == "__main__":
    client = VideoClient()
    client.run()
