import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch, torchvision
import torchvision.transforms as t


from utils import server_utils, annotation_utils

# https://github.com/hubert10/fasterrcnn_resnet50_fpn_v2_new_dataset/blob/main/datasets.py

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 annot_format: str,
                 classes: list[int] = None,
                 height: int = 1024,
                 width: int = 1024,
                 use_train: bool = True,
                 use_test: bool = False,
                 use_test_raw: bool = False,
                 use_val: bool = False,
                 max_dataset_length: int = None,
                 transforms=None,  # Not implemented yet
                 ):
        """

        :param dataset_path:
        :param annot_format: yoloV5, faster, ...
        :param classes: list of global label ids(keys) to be used in dataset
        :param height: target image height
        :param width: target image width
        :param use_train: whether to use train images
        :param use_test: whether to use test images
        :param use_test_raw: whether to use test_raw
        :param use_val: whether to use validation images
        :param max_dataset_length:
        :param transforms:
        """
        self.__dataset_path = dataset_path
        self.__annot_format = annot_format
        self.__classes = list(set(annotation_utils.AnnotationLabels.GLOBAL_LABELS.values())) \
            if not classes else classes
        self.__height = height
        self.__width = width
        self.__use_train = use_train
        self.__use_test = use_test
        self.__use_test_raw = use_test_raw
        self.__use_val = use_val
        self.__max_dataset_length = max_dataset_length
        self.__transforms = transforms

        self.__image_file_type = ".JPG"
        self.__annot_file_type = ".txt"
        self.__allowed__annot_formats = ["yoloV5", "faster"]

        self.__all_images = []
        self.__all_annots = []

        # Check that only 1 source is used
        if (self.__use_train + self.__use_test + self.__use_val + self.__use_test_raw) != 1:
            raise Exception(f"Only one source is allowed. Select train, test or validation images")
        if self.__use_train:
            images_folder = "train"
        elif self.__use_test:
            images_folder = "test"
        elif self.__use_test_raw:
            images_folder = "test_raw"
        elif self.__use_val:
            images_folder = "val"
        else:
            raise Exception("Unknow image location")
        self.__images_path = os.path.join(self.__dataset_path, images_folder)

        # CHeck annot format
        if self.__annot_format not in self.__allowed__annot_formats:
            raise Exception(f"Selected annotation format '{self.__annot_format}' invalid")

        # Remove all annotations and images when no object is present.
        self.__read_and_clean()

    def __read_and_clean(self):
        all_images_path = glob.glob(self.__images_path + "/*" + self.__image_file_type)
        self.__all_images = [image_path.split(os.path.sep)[-1] for image_path in all_images_path]

        self.__all_images.sort(key=lambda x: (int(x.removesuffix(self.__image_file_type))))

        for image_name in list(self.__all_images):
            annot_name = image_name.replace(self.__image_file_type, self.__annot_file_type)

            # Remove empty images
            annot_path = os.path.join(self.__images_path, annot_name)
            if os.path.getsize(annot_path) == 0:
                self.__all_images.remove(image_name)
                continue

            # Remove images without target classes
            annot_data = np.loadtxt(
                annot_path,
                dtype=int, ndmin=2, delimiter=","
            )
            classes = annot_data[:, 0]
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

        img = cv2.imread(os.path.join(self.__images_path, image_name))
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, _ = img.shape

        annot_data = np.loadtxt(
            os.path.join(self.__images_path, annot_name),
            dtype=int, ndmin=2, delimiter=","
        )

        classes = annot_data[:, 0]
        valid_annots = [cl in self.__classes for cl in classes]
        not_valid_annots_index = [i for i, x in enumerate(valid_annots) if not x]
        annot_data = np.delete(annot_data, not_valid_annots_index, axis=0)

        labels = annot_data[:, 0]
        annotations = annot_data[:, 1:5]

        if height != self.__height or width != self.__width:
            print("Scaling not implemented yet, contact with developer!")
            # raise Exception("Scaling not implemented yet, contact with developer")
            # Resize image
            # Scale annotations

        if self.__classes != list(set(annotation_utils.AnnotationLabels.GLOBAL_LABELS.values())):
            # This means that not all classes are selected, but for fasterCNN labels should be [1, ..]
            include_class_array = np.array(self.__classes).reshape(1, -1)
            for i, (label, annot) in enumerate(zip(labels, annotations)):
                labels[i] = self.__classes.index(label) + 1

        # print(f"labels: {labels}")
        # print(f"anns: {annotations}")
        # plt.imshow(img)
        # plt.show()

        target = {"labels": torch.tensor(labels, dtype=torch.int64),
                  "boxes": torch.tensor(annotations, dtype=torch.float)}
        return t.ToTensor()(img), target

#TODO: collate functions
def collate_fn(batch):
    """
    To handle the data loading as different images may have different number
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))


if __name__ == "__main__":
    dataset = CustomDataset(
        dataset_path="datasets/2024-09-18 11:32:03.928673",
        annot_format="faster",
        classes=[2],
        use_train = True,
        use_test = False,
        use_val = False,
        max_dataset_length=None
    )

    print(len(dataset))

    img, labels = dataset[1]
    img_orig = (img.permute(1, 2, 0).cpu().detach().numpy() * 255).astype("uint8")
    img_orig = np.ascontiguousarray(img_orig, dtype=np.uint8)

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



