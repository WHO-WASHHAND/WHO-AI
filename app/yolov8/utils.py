from typing import Tuple
from collections import Counter
import numpy as np
import cv2

class_names = ['step1','step2','step3','step4','step5','step6']

# Create a list of colors for each class where each color is a tuple of 3 integer values
rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(class_names), 3))


def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes

def multiclass_nms(boxes, scores, class_ids, iou_threshold):

    unique_class_ids = np.unique(class_ids)

    keep_boxes = []
    for class_id in unique_class_ids:
        class_indices = np.where(class_ids == class_id)[0]
        class_boxes = boxes[class_indices,:]
        class_scores = scores[class_indices]

        class_keep_boxes = nms(class_boxes, class_scores, iou_threshold)
        keep_boxes.extend(class_indices[class_keep_boxes])

    return keep_boxes

def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou




def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def draw_detections(image, boxes, scores, class_ids, mask_alpha=0.3):
    det_img = image.copy()
    img_height, img_width = image.shape[:2]
    font_size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)

    det_img = draw_masks(det_img, boxes, class_ids, mask_alpha)

    # Draw bounding boxes and labels of detections
    for class_id, box, score in zip(class_ids, boxes, scores):
        color = colors[class_id]
        draw_box(det_img, box, color)
        label = class_names[class_id]
        caption = f'{label} {int(score * 100)}%'
        draw_text(det_img, caption, box, color, font_size, text_thickness)

    return det_img


def draw_box( image: np.ndarray, box: np.ndarray, color: Tuple[int, int, int] = (0, 0, 255),
             thickness: int = 2) -> np.ndarray:
    box = xywh2xyxy((box))
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box.astype(int)
    x1 = max(0, min(x1, width - 1))  # Giới hạn x1 trong khoảng [0, width-1]
    y1 = max(0, min(y1, height - 1))  # Giới hạn y1 trong khoảng [0, height-1]
    x2 = max(0, min(x2, width - 1))  # Giới hạn x2 trong khoảng [0, width-1]
    y2 = max(0, min(y2, height - 1))
    return cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def draw_text(image: np.ndarray, text: str, box: np.ndarray, color: Tuple[int, int, int] = (0, 0, 255),
              font_size: float = 0.001, text_thickness: int = 2) -> np.ndarray:
    box = xywh2xyxy((box))
    x1, y1, x2, y2 = box.astype(int)
    (tw, th), _ = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                  fontScale=font_size, thickness=text_thickness)
    th = int(th * 1.2)
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box.astype(int)
    x1 = max(0, min(x1, width - 1))  # Giới hạn x1 trong khoảng [0, width-1]
    y1 = max(0, min(y1, height - 1))  # Giới hạn y1 trong khoảng [0, height-1]
    x2 = max(0, min(x2, width - 1))  # Giới hạn x2 trong khoảng [0, width-1]
    y2 = max(0, min(y2, height - 1))
    cv2.rectangle(image, (x1, y1),
                  (x1 + tw, y1 - th), color, -1)

    return cv2.putText(image, text, (x1 + 10, y1 - 10 if y1 - 10 > 10 else y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), text_thickness, cv2.LINE_AA)

def draw_masks(image: np.ndarray, boxes: np.ndarray, classes: np.ndarray, mask_alpha: float = 0.3) -> np.ndarray:
    mask_img = image.copy()

    # Draw bounding boxes and labels of detections
    for box, class_id in zip(boxes, classes):
        color = colors[class_id]
        box = xywh2xyxy((box))
        x1, y1, x2, y2 = box.astype(int)
        # Draw fill rectangle in mask image
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

    return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)

def process_steps_with_threshold(data, time_threshold=50):
    # Kết quả lưu các step đã hoàn thành
    completed_steps = []

    # Biến lưu trữ thông tin về step hiện tại
    step_window = []
    start_time = None

    for i, (step, time) in enumerate(data):
        if start_time is None:
            start_time = time  # Khởi tạo thời gian bắt đầu cho window

        # Thêm step vào cửa sổ theo dõi
        step_window.append((step, time))

        # Kiểm tra nếu đã đạt ngưỡng thời gian
        if time - start_time >= time_threshold or i == len(data) - 1:
            # Tính tần suất xuất hiện của các step trong cửa sổ
            step_counts = Counter([s for s, _ in step_window])
            most_common_step, count = step_counts.most_common(1)[0]

            # Ghi lại step có tần suất xuất hiện nhiều nhất trong cửa sổ
            if most_common_step is not None:
                step_start_time = step_window[0][1]
                step_end_time = step_window[-1][1]
                completed_steps.append([most_common_step, step_start_time, step_end_time])

            # Reset lại cửa sổ theo dõi cho các step tiếp theo
            step_window = []
            start_time = None

    return completed_steps

def frame_to_time(frame, fps):
    total_seconds = frame / fps
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    return f"{minutes:02}:{seconds:02}"