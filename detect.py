from datasets import FasterCnnDataset, collate_fn
from model import create_model

import numpy as np
import torch, torchvision
import os
import cv2 as cv
import matplotlib.pyplot as plt
import argparse

class CustomModelDetect:
    def __init__(self, model_weights):

        self.num_classes = 2
        # self.model = create_model(self.num_classes).to("cpu")
        self.model = torch.load(model_weights).to("cpu")

    def inference(self, img):
        self.model.eval()

        images = [img]
        out = self.model(images)
        return out



if __name__ == "__main__":
    #TODO: create and read yaml file from dataset ??
    #TODO: do label mapping
    #TODO: add argparser

    output_dir = "./yolo_vs_faster"
    yolo_weights = "./weights/priede_10_05_24_4070546.pt"
    faster_weights = "./weights/priede_faster_epoch50.pth"
    image_source = "/home/hercogs/Desktop/Droni/git_repos/forestai_datasets_manager/datasets/priede_16_10_24/images/test_raw"  # Single image or folder
    annotations_source = "/home/hercogs/Desktop/Droni/git_repos/forestai_datasets_manager/datasets/priede_16_10_24/labels/test_raw"  # Single image or folder

    device = "cuda" # cpu
    torch.cuda.empty_cache()

    cmi = CustomModelDetect("/home/hercogs/Desktop/Droni/git_repos/forestai_fasterCNN/run1/train/exp2/last.pth")

    train_dataset = FasterCnnDataset(
        dataset_path="/home/hercogs/Desktop/Droni/git_repos/forestai_datasets_manager/datasets/egle_16_10_24",
        dataset_subfolder="test",
        classes=None,
        max_dataset_length=None,
        use_empty_images=False
    )

    for ii in range(3):

        img, labels = train_dataset[ii]

        img_orig = (img.permute(1, 2, 0).cpu().detach().numpy() * 255)
        img_orig = np.ascontiguousarray(img_orig, dtype=np.uint8)

        output = cmi.inference(img.to("cpu"))
        out_boxes = output[0]["boxes"].cpu().detach()
        out_scores = output[0]["scores"].cpu().detach()
        keep = torchvision.ops.nms(out_boxes, out_scores, 0.45)
        # print(keep, out_boxes.shape)
        out_boxes = torch.index_select(out_boxes, 0, keep)
        out_scores = torch.index_select(out_scores, 0, keep)
        # print(keep, out_boxes.shape)

        print(out_scores)
        for box, score in zip(out_boxes, out_scores):
            if score < 0.45:
                continue
            box = box.numpy().astype("int")
            img_orig = cv.rectangle(
                img_orig,
                (box[0], box[1], box[2]-box[0], box[3]-box[1]),
                (255, 0, 0), #rgb
                7
            )

        orig_boxes = labels["boxes"]
        for box in orig_boxes:
            box = box.numpy().astype("int")
            img_orig = cv.rectangle(img_orig,
                            (box[0], box[1], box[2]-box[0], box[3]-box[1]),
                            (0, 255, 0),
                            7)

        plt.imshow(img_orig)
        plt.show()



