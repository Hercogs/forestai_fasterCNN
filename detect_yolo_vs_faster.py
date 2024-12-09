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
#
# weights = "/home/hercogs/Desktop/Droni/git_repos/forestai_fasterCNN/run1/train/exp8/last.pth"
# model = torch.load(weights, weights_only=False)
#
# detection_model = AutoDetectionModel.from_pretrained(
#    model_type='torchvision',
#    model=model, #Faster RCNN Model
#    confidence_threshold=0.7,
#    #image_size=8064, #Image's longest dimension
#    device="cpu", # or "cuda:0"
#    load_at_init=True,
# )
#
# img_path = '/home/hercogs/Desktop/Droni/git_repos/forestai_datasets_manager/datasets/egle_16_10_24/images/test_raw/13.JPG'
# label_path = '/home/hercogs/Desktop/Droni/git_repos/forestai_datasets_manager/datasets/egle_16_10_24/labels/test_raw/13.txt'
# img_filename_temp = img_path.split('/')[1]
# img_filename = img_filename_temp.split('.')[0]
#
# # print(img_filename)
# # img_pil = PIL.Image.open(img_path)
# # W, H = img_pil.size
# # print(W)
# s_h, s_w = 1024, 1024
# s_h, s_w = int(s_h), int(s_w)
#
# result = get_sliced_prediction(
#    img_path,
#    detection_model,
#    slice_height=s_h,
#    slice_width=s_w,
#    overlap_height_ratio=0.2,
#    overlap_width_ratio=0.2,
# )
#
# #print(result.object_prediction_list)
#
# result.export_visuals(
#    export_dir="./",
#    text_size=1,
#    rect_th=5,
# )
#
# def draw_bounding_boxes(image: np.array, object_prediction_list: list = None, labels: np.array = None, hide_labels=False) -> np.array:
#    # set rect_th for boxes
#    rect_th = 5
#    # set text_th for category names
#    text_th = 5
#    # set text_size for category names
#    text_size = 1
#
#    color_true = (0, 255, 0) # rgb
#    color_predicted = (255, 0, 0)
#
#    # add true bboxes to image if present
#    labels = labels or []
#
#    for label in labels:
#
#       category_name = f"coded_{label[0]}"
#       score = 0
#
#       # set bbox points
#       point1, point2 = (int(label[1]), int(label[2])), (int(label[3]), int(label[4]))
#       # visualize boxes
#       cv2.rectangle(
#          image,
#          point1,
#          point2,
#          color=color_true,
#          thickness=rect_th+3, # Make bigger, so
#       )
#
#       if not hide_labels:
#          # arange bounding box text location
#          label = f"{category_name}"
#
#          hide_conf = True
#          if not hide_conf:
#             label += f" {score:.2f}"
#
#          box_width, box_height = cv2.getTextSize(label, 0, fontScale=text_size, thickness=text_th)[
#             0
#          ]  # label width, height
#          outside = point1[1] - box_height - 3 >= 0  # label fits outside box
#          point2 = point1[0] + box_width, point1[1] - box_height - 3 if outside else point1[1] + box_height + 3
#          # add bounding box text
#          cv2.rectangle(image, point1, point2, color_true, -1, cv2.LINE_AA)  # filled
#          cv2.putText(
#             image,
#             label,
#             (point1[0], point1[1] - 2 if outside else point1[1] + box_height + 2),
#             0,
#             text_size,
#             (255, 255, 255),
#             thickness=text_th,
#          )
#
#    # add predicted bboxes to image if present
#    object_prediction_list = object_prediction_list or []
#
#    for object_prediction in object_prediction_list:
#       # deepcopy object_prediction_list so that original is not altered
#       object_prediction = object_prediction.deepcopy()
#
#       bbox = object_prediction.bbox.to_xyxy()
#       category_name = object_prediction.category.name
#       score = object_prediction.score.value
#
#       # set bbox points
#       point1, point2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
#       # visualize boxes
#       cv2.rectangle(
#          image,
#          point1,
#          point2,
#          color=color_predicted,
#          thickness=rect_th,
#       )
#
#       if not hide_labels:
#          # arange bounding box text location
#          label = f"{category_name}"
#
#          hide_conf = False
#          if not hide_conf:
#             label += f" {score:.2f}"
#
#          box_width, box_height = cv2.getTextSize(label, 0, fontScale=text_size, thickness=text_th)[
#             0
#          ]  # label width, height
#          outside = point1[1] - box_height - 3 >= 0  # label fits outside box
#          point2 = point1[0] + box_width, point1[1] - box_height - 3 if outside else point1[1] + box_height + 3
#          # add bounding box text
#          cv2.rectangle(image, point1, point2, color_predicted, -1, cv2.LINE_AA)  # filled
#          cv2.putText(
#             image,
#             label,
#             (point1[0], point1[1] - 2 if outside else point1[1] + box_height + 2),
#             0,
#             text_size,
#             (255, 255, 255),
#             thickness=text_th,
#          )
#    return image
#
# img = cv2.imread(img_path)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
# labels = np.loadtxt(
#    label_path,
#    dtype=float, ndmin=2, delimiter=" "
# )
# labels = labels.tolist()
#
# img = draw_bounding_boxes(image=img, object_prediction_list=result.object_prediction_list, labels=labels)
#
# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#
# out_dir = "./sliced_prediction"
# os.makedirs(out_dir, exist_ok=True)
# cv2.imwrite(f"{out_dir}/out.png", img)

def get_images_and_annotations(image_path: str, annotation_path: str) -> tuple[list[str], list[str]]:
   """
   This function returns tuple with images paths and annotations paths.
   Annotation format is expected to be same as in custom generated dataset.
   :param image_path: location of images or single image
   :param annotation_path: location of annotations or single annotation
   :return: tuple with list of images and list of annotations
   """
   images, annotations = ([], [])
   if os.path.isfile(image_path) and os.path.isfile(annotation_path):
      # Given single source
      img_name_full = os.path.basename(image_path)
      annot_name_full = os.path.basename(annotation_path)
      img_name, img_ext = os.path.splitext(img_name_full)
      annot_name, annot_ext = os.path.splitext(annot_name_full)
      if img_name != annot_name:
         raise Exception(f"Image name {img_name} not equal not annotation name {annot_name}.")
      images.append(image_path)
      annotations.append(annotation_path)
   else:
      # Given dir with sources
      # Iterate over every image in this dir
      for file in os.listdir(image_path):
         if file.lower().endswith((".jpg", ".jpeg", ".png")):
            # Check if annotation exists for this file
            img_name, img_ext = os.path.splitext(file)
            annot_path = os.path.join(annotation_path, file.replace(img_ext, ".txt"))
            if os.path.isfile(annot_path):
               images.append(os.path.join(image_path, file))
               annotations.append(annot_path)

   return images, annotations

def draw_ground_truth_bboxes(image: np.array, label_path: str, color_true, rect_th):
   cv2.line(image, (50, 50), (200, 50), color_true, 15)
   cv2.putText(
      image, "TRUE", (250, 75), 0, 2, color_true, thickness=10,
   )

   if os.stat(label_path).st_size == 0:
      return
   labels = np.loadtxt(
      label_path,
      dtype=float, ndmin=2, delimiter=" "
   )
   labels = labels.tolist()

   for label in labels:
      point1, point2 = (int(label[1]), int(label[2])), (int(label[3]), int(label[4]))
      # visualize boxes
      cv2.rectangle(
         image,
         point1,
         point2,
         color=color_true,
         thickness=rect_th+3, # Make bigger, so
      )

def draw_faster_cnn_bboxes(image: np.array, faster_result, color_predicted_faster, rect_th):
   cv2.line(image, (50, 100), (200, 100), color_predicted_faster, 15)
   cv2.putText(
      image, "FASTER", (250, 125), 0, 2, color_predicted_faster, thickness=10,
   )

   object_prediction_list = faster_result.object_prediction_list

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
         color=color_predicted_faster,
         thickness=rect_th,
      )

      hide_labels = False
      text_size = 1
      text_th = 3

      if not hide_labels:
         # arange bounding box text location
         label = f"{category_name}"

         hide_conf = False
         if not hide_conf:
            label += f" {score:.2f}"

         label = f"{score:.2f}"

         box_width, box_height = cv2.getTextSize(label, 0, fontScale=text_size, thickness=text_th)[
            0
         ]  # label width, height
         outside = point1[1] - box_height - 3 >= 0  # label fits outside box
         point2 = point1[0] + box_width, point1[1] - box_height - 3 if outside else point1[1] + box_height + 3
         # add bounding box text
         cv2.rectangle(image, point1, point2, color_predicted_faster, -1, cv2.LINE_AA)  # filled
         cv2.putText(
            image,
            label,
            (point1[0], point1[1] - 2 if outside else point1[1] + box_height + 2),
            0,
            text_size,
            (255, 255, 255),
            thickness=text_th,
         )

def draw_yolo_bboxes(image: np.array, yolo_result, color_predicted_yolo, rect_th):
   cv2.line(image, (50, 150), (200, 150), color_predicted_yolo, 15)
   cv2.putText(
      image, "YOLO", (250, 175), 0, 2, color_predicted_yolo, thickness=10,
   )

   object_prediction_list = yolo_result.object_prediction_list

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
         color=color_predicted_yolo,
         thickness=rect_th-2,
      )

      hide_labels = False
      text_size = 1
      text_th = 3

      if not hide_labels:
         # arange bounding box text location
         label = f"{category_name}"

         hide_conf = False
         if not hide_conf:
            label += f" {score:.2f}"

         label = f"{score:.2f}"

         box_width, box_height = cv2.getTextSize(label, 0, fontScale=text_size, thickness=text_th)[
            0
         ]  # label width, height
         outside = point1[1] - box_height - 3 >= 0  # label fits outside box
         point2 = point1[0] + box_width, point1[1] - box_height - 3 if outside else point1[1] + box_height + 3
         # add bounding box text
         cv2.rectangle(image, point1, point2, color_predicted_yolo, -1, cv2.LINE_AA)  # filled
         cv2.putText(
            image,
            label,
            (point1[0], point1[1] - 2 if outside else point1[1] + box_height + 2),
            0,
            text_size,
            (255, 255, 255),
            thickness=text_th,
         )


def main():
   output_dir = "./yolo_vs_faster"
   yolo_weights = "./weights/yolo_priede_all.pt"
   faster_weights = "./weights/priede_faster_best.pth"
   image_source = "/home/hercogs/Desktop/Droni/git_repos/forestai_datasets_manager/datasets/priede_16_10_24/images/test_raw"  # Single image or folder
   annotations_source = "/home/hercogs/Desktop/Droni/git_repos/forestai_datasets_manager/datasets/priede_16_10_24/labels/test_raw"  # Single image or folder
   device = "cpu"

   rect_th = 8  # set rect_th for boxes
   text_th = 5  # set text_th for category names
   text_size = 1  # set text_size for category names

   color_true = (0, 255, 0) # BGR -> green
   color_predicted_yolo = (0, 0, 255) # BGR -> blue
   color_predicted_faster = (252, 15, 192) # BGR -> pink
   #color_predicted_faster = (255, 255, 0)  # BGR -> pink

   # Create fasterCNN model
   faster_cnn_model = torch.load(faster_weights, weights_only=False)
   faster_cnn_detection_model = AutoDetectionModel.from_pretrained(
      model_type='torchvision',
      model=faster_cnn_model, #Faster RCNN Model
      confidence_threshold=0.4,
      device=device, # or "cuda:0"
      load_at_init=True,
   )

   # Create yolo model
   yolo_detection_model = AutoDetectionModel.from_pretrained(
      model_type='yolov5',
      model_path=yolo_weights,
      confidence_threshold=0.4,
      device=device,  # or 'cuda:0'
      load_at_init=True,
   )

   os.makedirs(output_dir, exist_ok=True)  # Create output dir

   images, labels = get_images_and_annotations(image_source, annotations_source)

   for idx, (image, label) in enumerate(zip(images, labels)):
      print(f"Processing image {idx+1} out of {len(images)}: {image}")

      img = cv2.imread(image)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

      # Get Yolo prediction object list
      yolo_result = get_sliced_prediction(
         image,
         yolo_detection_model,
         slice_height=1024,
         slice_width=1024,
         overlap_height_ratio=0.2,
         overlap_width_ratio=0.2,
      )

      # Get FasterCNN prediction object list
      faster_result = get_sliced_prediction(
         image,
         faster_cnn_detection_model,
         slice_height=1024,
         slice_width=1024,
         overlap_height_ratio=0.2,
         overlap_width_ratio=0.2,
      )

      draw_ground_truth_bboxes(img, label, color_true, rect_th)
      draw_faster_cnn_bboxes(img, faster_result, color_predicted_faster, rect_th)
      draw_yolo_bboxes(img, yolo_result, color_predicted_yolo, rect_th)


      img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
      cv2.imwrite(f"{output_dir}/{idx+1}.JPG", img)


if __name__ == "__main__":
   main()
