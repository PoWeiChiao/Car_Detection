import cv2 as cv
import glob
import os
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import transforms
from model import SSD300
from utils.utils import *

def detect(net, device, image_path, min_score, max_overlap, top_k):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.
    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    # Transform
    original_image = Image.open(image_path)
    image_transforms = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = image_transforms(original_image)
    image = image.unsqueeze(0)
    image = image.to(device)

    predicted_locs, predicted_scores = net(image)

    det_boxes, det_labels, det_scores = net.detect_objects(predicted_locs, predicted_scores, min_score=min_score, max_overlap=max_overlap, top_k=top_k)

    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor([original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    det_labels = det_labels[0].to('cpu').tolist()
    det_scores = det_scores[0].to('cpu').tolist()

    return det_boxes, det_labels, det_scores

def plotbbox(cv_image, det_boxes, det_labels, det_scores, threshold=0.5):
    if det_labels == ['background']:
        return cv_image
    for i in range(det_boxes.size(0)):
        if det_scores[i] <= threshold:
            continue
        box_location = det_boxes[i].tolist()
        xmin = int(box_location[0])
        ymin = int(box_location[1])
        xmax = int(box_location[2])
        ymax = int(box_location[3])
        label = det_labels[i]
        
        if label == 1:
            cv.rectangle(cv_image, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=2)
            cv.putText(cv_image, 'car', (xmin, ymin), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), thickness=2)
        elif label == 2:
            cv.rectangle(cv_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness=2)
            cv.putText(cv_image, 'truck', (xmin, ymin), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), thickness=2)
        elif label == 3:
            cv.rectangle(cv_image, (xmin, ymin), (xmax, ymax), (255, 0, 0), thickness=2)
            cv.putText(cv_image, 'bus', (xmin, ymin), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), thickness=2)
    return cv_image

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = SSD300(n_classes=4)
    net.load_state_dict(torch.load('saved/model_best.pth', map_location=device))
    net.to(device=device)
    
    test_dir = 'data/test'
    save_dir = 'predict'
    image_list = glob.glob(os.path.join(test_dir, 'image', '*.jpg'))
    for path in image_list:
        det_boxes, det_labels, det_scores = detect(net, device, path, min_score=0.2, max_overlap=0.5, top_k=200)
        cv_image = cv.imread(path)
        cv_image = plotbbox(cv_image, det_boxes, det_labels, det_scores, threshold=0.5)
        cv.imwrite(os.path.join(save_dir, os.path.basename(path)), cv_image)
        print(os.path.basename(path))

if __name__ == '__main__':
    main()
    