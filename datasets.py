import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch, torchvision
import torchvision.transforms as t

# https://github.com/hubert10/fasterrcnn_resnet50_fpn_v2_new_dataset/blob/main/datasets.py

class FasterCnnDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 dataset_subfolder: str,
                 classes: list[int] = None,
                 height: int = 1024,
                 width: int = 1024,
                 max_dataset_length: int = None,
                 use_empty_images: bool = False,
                 transforms=None,  # Not implemented yet
                 ):
        """

        :param dataset_path:
        :param dataset_subfolder: option of ["test", "test_raw", "train", "val"]
        :param classes: list of global label ids(keys) to be used in dataset
        :param height: target image height
        :param width: target image width
        :param max_dataset_length:
        :param use_empty_images: whether to use images with no annotations
        :param transforms:
        """
        self.__dataset_path = dataset_path
        self.__dataset_subfolder = dataset_subfolder
        self.__classes = [] if not classes else classes
        self.__height = height
        self.__width = width
        self.__max_dataset_length = max_dataset_length
        self.__use_empty_images = use_empty_images
        self.__transforms = transforms

        self.__image_file_type = ".JPG"
        self.__annot_file_type = ".txt"

        self.__all_images = []
        self.__all_annots = []

        self.__images_path = os.path.join(self.__dataset_path, "images", self.__dataset_subfolder)
        self.__labels_path = os.path.join(self.__dataset_path, "labels", self.__dataset_subfolder)

        if not os.path.exists(self.__images_path):
            raise Exception(f"Images paths do not exist: {self.__images_path}")
        if not os.path.exists(self.__labels_path):
            raise Exception(f"Labels paths do not exist: {self.__labels_path}")

        # Remove all annotations and images when no object is present.
        self.__read_and_clean()

    def __read_and_clean(self):
        all_images_path = glob.glob(self.__images_path + "/*" + self.__image_file_type)
        self.__all_images = [image_path.split(os.path.sep)[-1] for image_path in all_images_path]

        self.__all_images.sort(key=lambda x: (int(x.removesuffix(self.__image_file_type))))

        for image_name in list(self.__all_images):
            if not self.__use_empty_images:
                annot_name = image_name.replace(self.__image_file_type, self.__annot_file_type)

                # Remove empty images
                annot_path = os.path.join(self.__labels_path, annot_name)
                if os.path.getsize(annot_path) == 0:
                    self.__all_images.remove(image_name)
                    continue

                if not self.__classes:
                    # Use all classes
                    continue

                # Remove images without target classes
                annot_data = np.loadtxt(
                    annot_path,
                    dtype=float, ndmin=2, delimiter=" "
                )
                classes = annot_data[:, 0].astype(int)

                valid_annots = [cl in self.__classes for cl in classes]
                if not any(valid_annots):
                    self.__all_images.remove(image_name)

        if self.__max_dataset_length and len(self) > self.__max_dataset_length:
            self.__all_images = self.__all_images[:self.__max_dataset_length]

    def __len__(self):
        return len(self.__all_images)

    def __getitem__(self, idx):
        image_name = self.__all_images[idx]
        annot_name = image_name.replace(self.__image_file_type, self.__annot_file_type)

        image_path = os.path.join(self.__images_path, image_name)
        label_path = os.path.join(self.__labels_path, annot_name)


        img = cv2.imread(image_path)
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, _ = img.shape

        if height != self.__height or width != self.__width:
            #print("Scaling not implemented yet, contact with developer!")
            raise Exception("Scaling not implemented yet, contact with developer")
            # Resize image
            # Scale annotations

        annot_data = np.loadtxt(
            label_path,
            dtype=float, ndmin=2, delimiter=" "
        )

        """
        It is expected that dataset contains classes [0, 1, ... n]
        Faster CNN expects classes [1, 2, ... n+1]
        """

        annot_data_updated = np.empty((0, 5))

        for i, row in enumerate(np.copy(annot_data)):
            label = row[0]
            annotation = row[1:5]

            row_new = np.zeros(5)

            if self.__classes:
                if label not in self.__classes:
                    continue
                # Update label
                row_new[0] = self.__classes.index(label) + 1
            else:
                # Update label
                row_new[0] = label + 1

            # Convert yoloV5 to fasterCNN format
            # [class_id center_x center_y width height] -> [class_id, x_min, y_min, x_max, y_max]
            row_new[1] = annotation[0] * width - annotation[2] * width / 2
            row_new[2] = annotation[1] * height - annotation[3] * height / 2
            row_new[3] = annotation[0] * width + annotation[2] * width / 2
            row_new[4] = annotation[1] * height + annotation[3] * height / 2

            annot_data_updated = np.append(annot_data_updated, row_new.reshape(1, 5), axis=0)

        target = {"labels": torch.tensor(annot_data_updated[:, 0], dtype=torch.int64),
                  "boxes": torch.tensor(annot_data_updated[:, 1:5], dtype=torch.float)}
        return t.ToTensor()(img), target

#TODO: collate functions
def collate_fn(batch):
    """
    To handle the data loading as different images may have different number
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))


if __name__ == "__main__":
    dataset = FasterCnnDataset(
        dataset_path="/home/hercogs/Desktop/Droni/git_repos/forestai_dataset_generation/datasets/priede_test",
        dataset_subfolder="train",
        classes=None,
        max_dataset_length=None,
        use_empty_images=False
    )

    print(f"Length of dataset: {len(dataset)}")

    img, labels = dataset[1]
    img_orig = (img.permute(1, 2, 0).cpu().detach().numpy() * 255).astype("uint8")
    img_orig = np.ascontiguousarray(img_orig, dtype=np.uint8)

    print(f"Labels: {labels}")

    orig_boxes = labels["boxes"]
    for box in orig_boxes:
        box = box.numpy().astype("int")
        img_orig = cv2.rectangle(img_orig,
                       (box[0], box[1], box[2]-box[0], box[3]-box[1]),
                       (0, 255, 0),
                       7)

    plt.imshow(img_orig)
    plt.show()




    #
    # # print(labels["label"])
    #
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
    #     weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    # )
    # # Get the number of input features
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # # define a new head for the detector with required number of classes
    # model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)
    #
    # targets = [labels]
    # images = [img]
    # print(model(images, targets))
    #
    # model.eval()
    # print(model(images))



