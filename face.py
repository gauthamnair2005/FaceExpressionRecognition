import cv2
import numpy as np
import time
import os

from cv2 import dnn
from math import ceil

image_mean = np.array([127, 127, 127])
image_std = 128.0
iou_threshold = 0.3
center_variance = 0.1
size_variance = 0.2
min_boxes = [
    [10.0, 16.0, 24.0], 
    [32.0, 48.0], 
    [64.0, 96.0], 
    [128.0, 192.0, 256.0]
]
strides = [8.0, 16.0, 32.0, 64.0]
threshold = 0.5

def define_img_size(image_size):
    shrinkage_list = []
    feature_map_w_h_list = []
    for size in image_size:
        feature_map = [int(ceil(size / stride)) for stride in strides]
        feature_map_w_h_list.append(feature_map)

    for i in range(0, len(image_size)):
        shrinkage_list.append(strides)
    priors = generate_priors(
        feature_map_w_h_list, shrinkage_list, image_size, min_boxes
    )
    return priors


def generate_priors(
    feature_map_list, shrinkage_list, image_size, min_boxes
):
    priors = []
    for index in range(0, len(feature_map_list[0])):
        scale_w = image_size[0] / shrinkage_list[0][index]
        scale_h = image_size[1] / shrinkage_list[1][index]
        for j in range(0, feature_map_list[1][index]):
            for i in range(0, feature_map_list[0][index]):
                x_center = (i + 0.5) / scale_w
                y_center = (j + 0.5) / scale_h

                for min_box in min_boxes[index]:
                    w = min_box / image_size[0]
                    h = min_box / image_size[1]
                    priors.append([
                        x_center,
                        y_center,
                        w,
                        h
                    ])
    return np.clip(priors, 0.0, 1.0)


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]
    return box_scores[picked, :]


def area_of(left_top, right_bottom):
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:]) 

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def predict(
    width, 
    height, 
    confidences, 
    boxes, 
    prob_threshold, 
    iou_threshold=0.3, 
    top_k=-1
):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate(
            [subset_boxes, probs.reshape(-1, 1)], axis=1
        )
        box_probs = hard_nms(box_probs,
                             iou_threshold=iou_threshold,
                             top_k=top_k,
                             )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return (
        picked_box_probs[:, :4].astype(np.int32), 
        np.array(picked_labels), 
        picked_box_probs[:, 4]
    )


def convert_locations_to_boxes(locations, priors, center_variance,
                               size_variance):
    if len(priors.shape) + 1 == len(locations.shape):
        priors = np.expand_dims(priors, 0)
    return np.concatenate([
        locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
        np.exp(locations[..., 2:] * size_variance) * priors[..., 2:]
    ], axis=len(locations.shape) - 1)


def center_form_to_corner_form(locations):
    return np.concatenate(
        [locations[..., :2] - locations[..., 2:] / 2,
         locations[..., :2] + locations[..., 2:] / 2], 
        len(locations.shape) - 1
    )


def FER_live_cam():
    emotion_dict = {
        0: 'Neutral', 
        1: 'Happy', 
        2: 'Surprise', 
        3: 'Sad',
        4: 'Angry', 
        5: 'Disgust', 
        6: 'Fear'
    }

    cap = cv2.VideoCapture(0)  # Use camera 0 (default camera)
    print("Started...")

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    print("Recording Video.. Video")
    result = cv2.VideoWriter('FERRecording.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)

    # Read ONNX model
    model = 'onnx_model.onnx'
    model = cv2.dnn.readNetFromONNX('emotion-ferplus-8.onnx')
    
    # Read the Caffe face detector.
    model_path = 'RFB-320/RFB-320.caffemodel'
    proto_path = 'RFB-320/RFB-320.prototxt'
    net = dnn.readNetFromCaffe(proto_path, model_path)
    input_size = [320, 240]
    width = input_size[0]
    height = input_size[1]
    priors = define_img_size(input_size)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            img_ori = frame
            #print("frame size: ", frame.shape)
            rect = cv2.resize(img_ori, (width, height))
            rect = cv2.cvtColor(rect, cv2.COLOR_BGR2RGB)
            net.setInput(dnn.blobFromImage(
                rect, 1 / image_std, (width, height), 127)
            )
            start_time = time.time()
            boxes, scores = net.forward(["boxes", "scores"])
            boxes = np.expand_dims(np.reshape(boxes, (-1, 4)), axis=0)
            scores = np.expand_dims(np.reshape(scores, (-1, 2)), axis=0)
            boxes = convert_locations_to_boxes(
                boxes, priors, center_variance, size_variance
            )
            boxes = center_form_to_corner_form(boxes)
            boxes, labels, probs = predict(
                img_ori.shape[1], 
                img_ori.shape[0], 
                scores, 
                boxes, 
                threshold
            )
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for (x1, y1, x2, y2) in boxes:
                w = x2 - x1
                h = y2 - y1
                cv2.rectangle(frame, (x1,y1), (x2, y2), (0,255,0), 2)
                try:
                    resize_frame = cv2.resize(gray[y1:y1 + h, x1:x1 + w], (64, 64))
                    resize_frame = resize_frame.reshape(1, 1, 64, 64)
                    model.setInput(resize_frame)
                    output = model.forward()
                    end_time = time.time()
                    fps = 1 / (end_time - start_time)
                    pred = emotion_dict[list(output[0]).index(max(output[0]))]
                    cv2.rectangle(
                        img_ori, 
                        (x1, y1), 
                        (x2, y2), 
                        (0, 255, 0), 
                        2,
                        lineType=cv2.LINE_AA
                    )
                    cv2.putText(
                        frame, 
                        pred, 
                        (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, 
                        (0, 255, 0), 
                        2,
                        lineType=cv2.LINE_AA
                    )
                except cv2.error as e:
                    print("Error occurred: ", e)
                except Exception as e:
                    print("An error occurred: ", e)

            result.write(frame)
        
            cv2.namedWindow('Facial Expression Mini-Project by Gautham', cv2.WINDOW_NORMAL)
            cv2.setWindowProperty('Facial Expression Mini-Project by Gautham', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('Facial Expression Mini-Project by Gautham', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            break

    cap.release()
    result.release()
    cv2.destroyAllWindows()

    cap.release()
    result.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Facial Expression Recognition Mini-Project by Gautham Nair")
    print("Starting... (Press Ctrl + C to stop or press ESC to stop the recognition process)")
    FER_live_cam()
    print("Saved Video...")
    print("Exiting...")