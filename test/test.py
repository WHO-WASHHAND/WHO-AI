from collections import Counter

from app.yolov8.utils import frame_to_time

from collections import Counter

from collections import Counter


def process_steps_with_threshold(data, time_threshold=50, fps=25):
    completed_steps = []
    step_window = []
    start_time = None
    seen_steps = set()
    last_len_cs = len(completed_steps)
    # Set để theo dõi các step đã được thực hiện
    for i, (step, time) in enumerate(data):
        if start_time is None:
            start_time = time  # Khởi tạo thời gian bắt đầu cho window

        # Thêm step vào cửa sổ theo dõi
        step_window.append((step, time))

        # Kiểm tra nếu đã đạt ngưỡng thời gian hoặc nếu đó là phần tử cuối cùng
        if time - start_time >= time_threshold or i == len(data) - 1:
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

    print(completed_steps)
    return completed_steps


# Áp dụng vào dữ liệu
data = [[0, 125], [0, 150], [5, 175], [5, 200], [0, 225], [0, 250], [0, 275], [0, 350],
        [3, 375], [1, 400], [1, 425], [1, 450], [1, 550], [2, 625], [2, 650], [2, 675],
        [2, 700], [5, 725], [3, 750], [5, 775], [3, 800], [3, 825], [3, 850], [3, 875],
        [3, 900], [3, 925], [4, 950], [3, 1025], [4, 1050], [4, 1075], [4, 1100],
        [4, 1125], [4, 1150], [4, 1175], [5, 1225], [5, 1250], [5, 1275], [5, 1300],
        [5, 1325], [5, 1350], [5, 1375], [5, 1400], [5, 1425], [5, 1450], [5, 1475],
        [5, 1500], [5, 1525], [5, 1550], [5, 1575], [5, 1600], [5, 1625], [5, 1650],
        [5, 1675], [5, 1700], [5, 1800]]
data1 = []
time_threshold = 100

for i in range(len(data)):
    data1.append(data[i])
    process_steps_with_threshold(data1, time_threshold, 25)
  # Ngưỡng thời gian cho mỗi step
#completed_steps = process_steps_with_threshold(data, time_threshold,25)


