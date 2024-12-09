from model import create_model

import numpy as np
import torch, torchvision
import os
import cv2
import matplotlib.pyplot as plt
import argparse


from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, predict, get_prediction
from sahi.utils.file import download_from_url
from sahi.prediction import visualize_object_predictions
from sahi.utils.cv import read_image

weights = "/home/hercogs/Desktop/Droni/git_repos/forestai_fasterCNN/run1/train/exp8/last.pth"
model = torch.load(weights, weights_only=False)

detection_model = AutoDetectionModel.from_pretrained(
   model_type='torchvision',
   model=model, #Faster RCNN Model
   confidence_threshold=0.7,
   #image_size=8064, #Image's longest dimension
   device="cpu", # or "cuda:0"
   load_at_init=True,
)

img_path = '/home/hercogs/Desktop/Droni/git_repos/forestai_datasets_manager/datasets/egle_16_10_24/images/test_raw/13.JPG'
label_path = '/home/hercogs/Desktop/Droni/git_repos/forestai_datasets_manager/datasets/egle_16_10_24/labels/test_raw/13.txt'
img_filename_temp = img_path.split('/')[1]
img_filename = img_filename_temp.split('.')[0]

# print(img_filename)
# img_pil = PIL.Image.open(img_path)
# W, H = img_pil.size
# print(W)
s_h, s_w = 1024, 1024
s_h, s_w = int(s_h), int(s_w)

result = get_sliced_prediction(
   img_path,
   detection_model,
   slice_height=s_h,
   slice_width=s_w,
   overlap_height_ratio=0.2,
   overlap_width_ratio=0.2,
)

#print(result.object_prediction_list)

result.export_visuals(
   export_dir="./",
   text_size=1,
   rect_th=5,
)

def draw_bounding_boxes(image: np.array, object_prediction_list: list = None, labels: np.array = None, hide_labels=False) -> np.array:
   # set rect_th for boxes
   rect_th = 5
   # set text_th for category names
   text_th = 5
   # set text_size for category names
   text_size = 1

   color_true = (0, 255, 0) # rgb
   color_predicted = (255, 0, 0)

   # add true bboxes to image if present
   labels = labels or []

   for label in labels:

      category_name = f"coded_{label[0]}"
      score = 0

      # set bbox points
      point1, point2 = (int(label[1]), int(label[2])), (int(label[3]), int(label[4]))
      # visualize boxes
      cv2.rectangle(
         image,
         point1,
         point2,
         color=color_true,
         thickness=rect_th+3, # Make bigger, so
      )

      if not hide_labels:
         # arange bounding box text location
         label = f"{category_name}"

         hide_conf = True
         if not hide_conf:
            label += f" {score:.2f}"

         box_width, box_height = cv2.getTextSize(label, 0, fontScale=text_size, thickness=text_th)[
            0
         ]  # label width, height
         outside = point1[1] - box_height - 3 >= 0  # label fits outside box
         point2 = point1[0] + box_width, point1[1] - box_height - 3 if outside else point1[1] + box_height + 3
         # add bounding box text
         cv2.rectangle(image, point1, point2, color_true, -1, cv2.LINE_AA)  # filled
         cv2.putText(
            image,
            label,
            (point1[0], point1[1] - 2 if outside else point1[1] + box_height + 2),
            0,
            text_size,
            (255, 255, 255),
            thickness=text_th,
         )

   # add predicted bboxes to image if present
   object_prediction_list = object_prediction_list or []

   for object_prediction in object_prediction_list:
      # deepcopy object_prediction_list so that original is not altered
      object_prediction = object_prediction.deepcopy()

      bbox = object_prediction.bbox.to_xyxy()
      category_name = object_prediction.category.name
      score = object_prediction.score.value

      # set bbox points
      point1, point2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
      # visualize boxes
      cv2.rectangle(
         image,
         point1,
         point2,
         color=color_predicted,
         thickness=rect_th,
      )

      if not hide_labels:
         # arange bounding box text location
         label = f"{category_name}"

         hide_conf = False
         if not hide_conf:
            label += f" {score:.2f}"

         box_width, box_height = cv2.getTextSize(label, 0, fontScale=text_size, thickness=text_th)[
            0
         ]  # label width, height
         outside = point1[1] - box_height - 3 >= 0  # label fits outside box
         point2 = point1[0] + box_width, point1[1] - box_height - 3 if outside else point1[1] + box_height + 3
         # add bounding box text
         cv2.rectangle(image, point1, point2, color_predicted, -1, cv2.LINE_AA)  # filled
         cv2.putText(
            image,
            label,
            (point1[0], point1[1] - 2 if outside else point1[1] + box_height + 2),
            0,
            text_size,
            (255, 255, 255),
            thickness=text_th,
         )
   return image

img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

labels = np.loadtxt(
   label_path,
   dtype=float, ndmin=2, delimiter=" "
)
labels = labels.tolist()

img = draw_bounding_boxes(image=img, object_prediction_list=result.object_prediction_list, labels=labels)

img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

out_dir = "./sliced_prediction"
os.makedirs(out_dir, exist_ok=True)
cv2.imwrite(f"{out_dir}/out.png", img)

