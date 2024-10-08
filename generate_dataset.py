import logging, logging.config
import os, shutil
import sys
import zipfile
import tarfile
import datetime
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import argparse
import yaml

from utils import server_utils, annotation_utils


class DatasetGenerator:
    """
    This class creates datasets
    """
    def __init__(self,
                 ssh_key_path: str,
                 selected_annotations: list[annotation_utils.AnnotationData],
                 dataset_name:str ="no_name_dataset",
                 use_test_frame:bool =True,
                 label_ids:list[int] =None,
                 slice_size:int =1024,
                 overwrite_data:bool = True,
                 dataset_format=None,
                 ):
        """
        :param ssh_key_path: location of private ssh key for server access
        :param selected_annotations: list of annotation to be used for dataset
        :param dataset_name: name of dataset
        :param use_test_frame: if True, use test frame in building dataset
        :param label_ids: list of global label ids(keys) to be used in dataset
        :param slice_size: size os sliced image
        :param overwrite_data: boolean weather to re download images
        :param dataset_format: Not used yet
        """
        # Set logging
        logging.basicConfig(format="%(filename)s:%(levelname)s - %(message)s", level=logging.ERROR)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        self.server_manager = server_utils.ServerManager(ssh_key_path)

        self.__selected_annotations = selected_annotations
        self.__dataset_name = dataset_name
        self.__use_test_frame = use_test_frame
        self.__label_ids = list(set(annotation_utils.AnnotationLabels.GLOBAL_LABELS.values()))\
            if not label_ids else label_ids
        # Remove tets_label id
        if annotation_utils.AnnotationLabels.TEST_LABEL_ID in self.__label_ids:
            self.__label_ids.remove(annotation_utils.AnnotationLabels.TEST_LABEL_ID)
        if not all(elem in annotation_utils.AnnotationLabels.GLOBAL_LABELS.values() for elem in self.__label_ids):
            raise Exception("Could not find provided label ids in global labels")
        self.__slice_size = slice_size
        self.__overwrite_data = overwrite_data

        # Counter names for images
        self.__number_images_train = 0
        self.__number_images_val = 0
        self.__number_images_test = 0
        self.__number_images_test_raw = 0

        # counter for each label in these categories
        self.__train_label_cnt = dict(zip(self.__label_ids, [0]*len(self.__label_ids)))
        self.__val_label_cnt = dict(zip(self.__label_ids, [0] * len(self.__label_ids)))
        self.__test_label_cnt = dict(zip(self.__label_ids, [0] * len(self.__label_ids)))
        self.__test_raw_label_cnt = dict(zip(self.__label_ids, [0] * len(self.__label_ids)))

        # Create dataset output folder structure
        self.__dataset_path = os.path.join("datasets", self.__dataset_name)
        if os.path.exists(self.__dataset_path):
            if self.__dataset_name != "test":
                raise Exception(f"Dataset with name '{self.__dataset_name}' already exists. Delete it or rename")
            else:
                # Delete test dataset
                shutil.rmtree(self.__dataset_path)

        self.__train_folder = os.path.join(self.__dataset_path, "train")  # Cropped images
        self.__val_folder = os.path.join(self.__dataset_path, "val")  # Cropped images
        self.__test_folder = os.path.join(self.__dataset_path, "test")  # Cropped images
        self.__test_raw_folder = os.path.join(self.__dataset_path, "test_raw")  # Original images with test_frame label

        self.logger.info(f"Generating dataset: {self.__dataset_path}")
        self.logger.info(f"Used label ids: {self.__label_ids}")
        self.logger.info(f"Used annotation jobs: {[ann.name for ann in self.__selected_annotations]}")

    def download_images(self, raw_data_location:str =None):
        """
        This function allows to download images from server which are used in annotations.
        :param raw_data_location: path or raw data location
        :return:
        """
        # There are to 2 options. Download all archive with images(not implemented) or download single images from server(implemeneted)
        #TODO: compress single images on server in archive, download it, unzip, delete archive from server
        #images = annotation.get_annotated_image_paths()
        # for i in images:
        #     self.server_manager.read_from_server(
        #         os.path.join(os.path.dirname(annotation.yaml_config["images_locations"]), i),
        #         os.path.join("datasets", "raw_data_images", annotation.name)
        #     )
        # Download *tar archive

        raw_data_location_path = os.path.join(".", "raw_data_images") \
            if raw_data_location is None else raw_data_location
        if not os.path.exists(raw_data_location_path):
            os.mkdir(raw_data_location_path)

        for annotation in self.__selected_annotations:
            self.logger.info(f"Downloading annotation {annotation.name} images ...")
            annotation.images_path = os.path.join(raw_data_location_path, annotation.name)

            if os.path.exists(os.path.join(raw_data_location_path, annotation.name)) and not self.__overwrite_data:
                self.logger.info(f"{annotation.name} path already exists. Skipped downloading!")
                continue

            self.server_manager.read_from_server(
                annotation.yaml_config["images_locations"],
                os.path.join(raw_data_location_path, annotation.name),
                overwrite=self.__overwrite_data
            )

            archive_name = os.path.basename(annotation.yaml_config["images_locations"])

            if not os.path.exists(os.path.join(raw_data_location_path, annotation.name, archive_name)):
                raise Exception(
                    f"Archive {os.path.join(raw_data_location_path, annotation.name, archive_name)} not exist!"
                )
            # Extract archive, if it exists
            self.logger.info(f"Extracting annotation {annotation.name} images ...")
            with tarfile.open(os.path.join(raw_data_location_path, annotation.name, archive_name),
                              "r") as tar_ref:
                tar_ref.extractall(path=os.path.join(raw_data_location_path, annotation.name))

    def create_dataset(self, dataset_format=None):
        """
        This functions creates dataset inside ./datasets folder
        :param dataset_format: resnet, yolo, whatever TODO: not implemented
        :return:
        """

        os.makedirs(self.__train_folder, exist_ok=True)  # Cropped images
        os.makedirs(self.__val_folder, exist_ok=True)  # Cropped images
        os.makedirs(self.__test_folder, exist_ok=True)  # Cropped images
        os.makedirs(self.__test_raw_folder, exist_ok=True)  # Original images with test_frame label

        #TODO: create *yaml file for dataset

        for idx, annotation in enumerate(self.__selected_annotations):
            self.logger.info(f"Job {idx+1} out of {len(self.__selected_annotations)} started")
            # TODO: set counter how many images are done

            # Iterate over every selected annotation objects
            for file in sorted(os.listdir(annotation.annotations_path)):
                # Iterate over single image in annotation

                if not os.path.getsize(os.path.join(annotation.annotations_path, file)) > 0:
                    # Skip empty annotations
                    continue
                # Read file content in numpy array
                yolo1_annotations = np.loadtxt(os.path.join(annotation.annotations_path, file), dtype=np.float64, ndmin=2)

                is_test_image = self.contains_test_label(yolo1_annotations, annotation.test_label_id)
                if is_test_image and not self.__use_test_frame:
                    # Skip annotation file if not use test_frame
                    continue


                img_path = os.path.join(annotation.images_path, file).replace(".txt", ".JPG")
                img = cv2.imread(img_path)
                image_height, image_width, _ = img.shape

                sliced_boxes = self.get_sliced_boxes(
                    image_height,
                    image_width,
                    slice_height=self.__slice_size,
                    slice_width=self.__slice_size,
                )

                fastercnn_annotations = self.convert_annotation_format(
                    image_height,
                    image_width,
                    yolo1_annotations
                )

                # Remap labels
                self.remap_labels(fastercnn_annotations, annotation)

                # Lambda function for checking if label ids are in use
                label_in_use = lambda label_ids: np.array([label_id in self.__label_ids for label_id in label_ids])

                # Delete test label
                fastercnn_annotations_filtered = np.delete(fastercnn_annotations,
                   np.where(fastercnn_annotations[:,0] == annotation_utils.AnnotationLabels.TEST_LABEL_ID),
                   axis=0)
                # Delete other labels which are not used
                fastercnn_annotations_filtered = np.delete(fastercnn_annotations_filtered,
                   np.where(np.invert(label_in_use(fastercnn_annotations_filtered[:, 0]))),
                   axis=0)

                for sliced_box in sliced_boxes:
                    # Iterate over every sliced box
                    sliced_annotation_list: list = []  # List to keep annotations for specific slice
                    for fastercnn_annotation in fastercnn_annotations_filtered:
                        if not self.annotation_inside_slice(sliced_box, fastercnn_annotation):
                            continue
                        if not (self.calculate_iou(sliced_box, fastercnn_annotation) > 0.7):
                            continue
                        # if label_in_use(np.array([fastercnn_annotation[0]])):
                        sliced_annotation_list.append(fastercnn_annotation)

                    # Convert global annotation frame to local frame
                    for i, sliced_annotation in enumerate(sliced_annotation_list):
                        sliced_annotation_list[i] = [
                            sliced_annotation[0],
                            max(0, sliced_annotation[1] - sliced_box[0]),
                            max(0, sliced_annotation[2] - sliced_box[1]),
                            min(self.__slice_size, sliced_annotation[3] - sliced_box[0]),
                            min(self.__slice_size, sliced_annotation[4] - sliced_box[1]),
                        ]

                    crop = img[sliced_box[1]:sliced_box[3], sliced_box[0]:sliced_box[2]]

                    if is_test_image:
                        self.__number_images_test += 1
                        image_name = str(self.__number_images_test)
                        destination_folder = self.__test_folder
                        self.update_label_count(self.__test_label_cnt, sliced_annotation_list)
                    else:
                        self.__number_images_train += 1
                        image_name = str(self.__number_images_train)
                        destination_folder = self.__train_folder
                        self.update_label_count(self.__train_label_cnt, sliced_annotation_list)

                    # Save cropped image
                    cv2.imwrite(os.path.join(destination_folder, image_name + ".JPG"), crop)
                    # Save *txt annotation file
                    np.savetxt(
                        fname=os.path.join(destination_folder, image_name + ".txt"),
                        X=np.array(sliced_annotation_list).astype(int),
                        fmt="%s",
                        delimiter=","
                    )

                if is_test_image:
                    # self.logger.info(f"Processing test_raw image: {img_path}")
                    # Save full image for later test purpose
                    self.__number_images_test_raw += 1
                    shutil.copy(img_path,
                        os.path.join(self.__test_raw_folder, str(self.__number_images_test_raw) + ".JPG"))
                    # # Delete test label
                    # fastercnn_annotations_filtered = np.delete(fastercnn_annotations,
                    #     np.where(fastercnn_annotations[:, 0] == annotation_utils.AnnotationLabels.TEST_LABEL_ID),
                    #     axis=0)
                    # # Delete other labels which are not used
                    # fastercnn_annotations_filtered = np.delete(fastercnn_annotations_filtered,
                    #     np.where(np.invert(label_in_use(fastercnn_annotations_filtered[:, 0]))), axis=0)
                    np.savetxt(
                        fname=os.path.join(self.__test_raw_folder, str(self.__number_images_test_raw) + ".txt"),
                        X=fastercnn_annotations_filtered.astype(int),
                        fmt="%s",
                        delimiter=","
                    )
                    # TODO: count how many labels are in train, test, test_raw
                    self.update_label_count(self.__test_raw_label_cnt, fastercnn_annotations_filtered)

        print(f"In total:\n"
              f"\ttrain images:\t\t",   f"{self.__number_images_train}\n"
              f"\ttest images:\t\t",    f"{self.__number_images_test}\n"
              f"\ttest_raw images:\t",  f"{self.__number_images_test_raw}\n"
        )

        print(f"self.__test_raw_label_cnt: \n{self.__test_raw_label_cnt}\n")
        print(f"self.__test_label_cnt: \n{self.__test_label_cnt}\n")
        print(f"self.__train_label_cnt: \n{self.__train_label_cnt}\n")
        # TODO: create same dataset meta info file

        return True

    def create_dataset_yolov5(self, dataset_format=None):
        """
        This functions creates dataset inside ./datasets folder in yoloV5 format
        :param dataset_format: resnet, yolo, whatever TODO: not implemented
        :return:
        """

        images_train_path = os.path.join(self.__dataset_path, "images", "train")
        images_val_path = os.path.join(self.__dataset_path, "images", "val")
        images_test_path = os.path.join(self.__dataset_path, "images", "test")
        images_test_raw_path = os.path.join(self.__dataset_path, "images", "test_raw")
        labels_train_path = os.path.join(self.__dataset_path, "labels", "train")
        labels_val_path = os.path.join(self.__dataset_path, "labels", "val")
        labels_test_path = os.path.join(self.__dataset_path, "labels", "test")
        labels_test_raw_path = os.path.join(self.__dataset_path, "labels", "test_raw")

        folders = [images_train_path, images_val_path, images_test_path, images_test_raw_path, \
                   labels_train_path, labels_val_path, labels_test_path, labels_test_raw_path]
        for f in folders:
            os.makedirs(f, exist_ok=True)

        # create {dataset_name}*yaml file for dataset in yoloV5
        with open(os.path.join(self.__dataset_path, f"{self.__dataset_name}.yaml"), "w") as f:
            #f.write(f'path: {os.path.abspath(self.__dataset_path)}/images\n')
            f.write(f'path: ./../{self.__dataset_name}/images\n')
            f.write('train: train\n')
            f.write('test: test\n')
            f.write('val: val\n')

            f.write('\n# number of classes\n')
            f.write(f'nc: {len(self.__label_ids)}\n')

            f.write('\n# class name\n')
            f.write('names:\n')
            for label_id in self.__label_ids:
                if self.__label_ids != list(set(annotation_utils.AnnotationLabels.GLOBAL_LABELS.values())):
                    # This means that not all classes are selected, but for fasterCNN labels should be [1, ..]
                    label_id_updated = self.__label_ids.index(label_id)
                else:
                    label_id_updated = label_id

                idx = list(annotation_utils.AnnotationLabels.GLOBAL_LABELS.values()).index(label_id)
                f.write(f"  {label_id_updated}: {list(annotation_utils.AnnotationLabels.GLOBAL_LABELS.keys())[idx]}\n")

        # Create dict for dataset meta info
        dataset_meta_info = {"used_annotation_jobs": {}}

        for idx, annotation in enumerate(self.__selected_annotations):
            self.logger.info(f"Job {idx+1} out of {len(self.__selected_annotations)} started")
            # TODO: set counter how many images are done

            dataset_meta_info["used_annotation_jobs"][idx+1] = {
                "annot_file": annotation.name,
                "info": {
                    "datums": annotation.datums,
                    "kadastrs": annotation.kadastrs,
                    "kvartals": annotation.kvartals,
                    "nogabals": annotation.nogabals,
                    "uav": annotation.uav
                }
            }

            # Iterate over every selected annotation objects
            for file in sorted(os.listdir(annotation.annotations_path)):
                # Iterate over single image in annotation

                if not os.path.getsize(os.path.join(annotation.annotations_path, file)) > 0:
                    # Skip empty annotations
                    continue
                # Read file content in numpy array
                yolo1_annotations = np.loadtxt(os.path.join(annotation.annotations_path, file), dtype=np.float64, ndmin=2)

                is_test_image = self.contains_test_label(yolo1_annotations, annotation.test_label_id)
                if is_test_image and not self.__use_test_frame:
                    # Skip annotation file if not use test_frame
                    continue


                img_path = os.path.join(annotation.images_path, file).replace(".txt", ".JPG")
                img = cv2.imread(img_path)
                image_height, image_width, _ = img.shape

                sliced_boxes = self.get_sliced_boxes(
                    image_height,
                    image_width,
                    slice_height=self.__slice_size,
                    slice_width=self.__slice_size,
                )

                fastercnn_annotations = self.convert_annotation_format(
                    image_height,
                    image_width,
                    yolo1_annotations
                )

                # Remap labels to global labels
                self.remap_labels(fastercnn_annotations, annotation)

                # Lambda function for checking if label ids are in use
                label_in_use = lambda label_ids: np.array([label_id in self.__label_ids for label_id in label_ids], dtype=bool)

                # Delete test label
                fastercnn_annotations_filtered = np.delete(fastercnn_annotations,
                   np.where(fastercnn_annotations[:,0] == annotation_utils.AnnotationLabels.TEST_LABEL_ID),
                   axis=0)
                # Delete other labels which are not used
                fastercnn_annotations_filtered = np.delete(fastercnn_annotations_filtered,
                   np.where(np.invert(label_in_use(fastercnn_annotations_filtered[:, 0]))),
                   axis=0)

                for sliced_box in sliced_boxes:
                    # Iterate over every sliced box
                    sliced_annotation_list: list = []  # List to keep annotations for specific slice
                    for fastercnn_annotation in fastercnn_annotations_filtered:
                        if not self.annotation_inside_slice(sliced_box, fastercnn_annotation):
                            continue
                        if not (self.calculate_iou(sliced_box, fastercnn_annotation) > 0.7):
                            continue
                        sliced_annotation_list.append(fastercnn_annotation)

                    # Convert global annotation frame to local frame
                    for i, sliced_annotation in enumerate(sliced_annotation_list):
                        sliced_annotation_list[i] = [
                            sliced_annotation[0],
                            max(0, sliced_annotation[1] - sliced_box[0]),
                            max(0, sliced_annotation[2] - sliced_box[1]),
                            min(self.__slice_size, sliced_annotation[3] - sliced_box[0]),
                            min(self.__slice_size, sliced_annotation[4] - sliced_box[1]),
                        ]

                    # Convert local px coordinates to yoloV5(relative) format
                    # class_id center_x center_y width height
                    for i, sliced_annotation in enumerate(sliced_annotation_list):
                        top_left_x = sliced_annotation[1] / self.__slice_size
                        top_left_y = sliced_annotation[2] / self.__slice_size
                        bot_right_x = sliced_annotation[3] / self.__slice_size
                        bot_right_y = sliced_annotation[4] / self.__slice_size

                        sliced_annotation_list[i] = [
                            sliced_annotation[0],
                            (top_left_x + bot_right_x) / 2,
                            (top_left_y + bot_right_y) / 2,
                            bot_right_x - top_left_x,
                            bot_right_y - top_left_y,
                        ]

                    crop = img[sliced_box[1]:sliced_box[3], sliced_box[0]:sliced_box[2]]

                    if is_test_image:
                        self.__number_images_test += 1
                        image_name = str(self.__number_images_test)
                        images_destination_folder = images_test_path
                        labels_destination_folder = labels_test_path
                        self.update_label_count(self.__test_label_cnt, sliced_annotation_list)

                    else:
                        # Split train and validation
                        if random.random() <= 0.15:
                            # Validation images
                            self.__number_images_val += 1
                            image_name = str(self.__number_images_val)
                            images_destination_folder = images_val_path
                            labels_destination_folder = labels_val_path
                            self.update_label_count(self.__val_label_cnt, sliced_annotation_list)
                        else:
                            # Training images
                            self.__number_images_train += 1
                            image_name = str(self.__number_images_train)
                            images_destination_folder = images_train_path
                            labels_destination_folder = labels_train_path
                            self.update_label_count(self.__train_label_cnt, sliced_annotation_list)

                    if self.__label_ids != list(set(annotation_utils.AnnotationLabels.GLOBAL_LABELS.values())):
                        # This means that not all classes are selected, but for dataset labels
                        # should start from 0 [0, 1, ..]
                        for i, sliced_annotation in enumerate(sliced_annotation_list):
                            sliced_annotation_list[i][0] = self.__label_ids.index(sliced_annotation[0])

                    # Save cropped image
                    cv2.imwrite(os.path.join(images_destination_folder, image_name + ".JPG"), crop)
                    # Save *txt annotation file
                    np.savetxt(
                        fname=os.path.join(labels_destination_folder, image_name + ".txt"),
                        X=np.array(sliced_annotation_list),
                        fmt="%s",
                        delimiter=" "
                    )

                if is_test_image:
                    # self.logger.info(f"Processing test_raw image: {img_path}")
                    # Save full image for later test purpose
                    self.__number_images_test_raw += 1
                    shutil.copy(img_path,
                        os.path.join(images_test_raw_path, str(self.__number_images_test_raw) + ".JPG"))

                    np.savetxt(
                        fname=os.path.join(labels_test_raw_path, str(self.__number_images_test_raw) + ".txt"),
                        X=fastercnn_annotations_filtered,
                        fmt="%s",
                        delimiter=" "
                    )
                    self.update_label_count(self.__test_raw_label_cnt, fastercnn_annotations_filtered)

        print(f"In total:\n"
              f"\ttrain images:\t\t",   f"{self.__number_images_train}\n"
              f"\tval images:\t\t",   f"{self.__number_images_val}\n" 
              f"\ttest images:\t\t",    f"{self.__number_images_test}\n"
              f"\ttest_raw images:\t",  f"{self.__number_images_test_raw}\n"
        )

        print(f"self.__test_raw_label_cnt: \n{self.__test_raw_label_cnt}\n")
        print(f"self.__test_label_cnt: \n{self.__test_label_cnt}\n")
        print(f"self.__train_label_cnt: \n{self.__train_label_cnt}\n")
        print(f"self.__val_label_cnt: \n{self.__val_label_cnt}\n")

        overall_dataset_meta_info = {
            "total_train_freq": {},
            "total_test_freq": {},
            "total_val_freq": {},
            "total_test_raw_freq": {},
        }
        for label_id, label_value in self.__train_label_cnt.items():
            if self.__label_ids != list(set(annotation_utils.AnnotationLabels.GLOBAL_LABELS.values())):
                # This means that not all classes are selected, but for fasterCNN labels should be [1, ..]
                label_id_updated = self.__label_ids.index(label_id)
            else:
                label_id_updated = label_id

            idx = list(annotation_utils.AnnotationLabels.GLOBAL_LABELS.values()).index(label_id)
            label_name = list(annotation_utils.AnnotationLabels.GLOBAL_LABELS.keys())[idx]
            overall_dataset_meta_info["total_train_freq"][label_id_updated] = f"{label_name} = {label_value}"
        for label_id, label_value in self.__test_label_cnt.items():
            if self.__label_ids != list(set(annotation_utils.AnnotationLabels.GLOBAL_LABELS.values())):
                # This means that not all classes are selected, but for fasterCNN labels should be [1, ..]
                label_id_updated = self.__label_ids.index(label_id)
            else:
                label_id_updated = label_id

            idx = list(annotation_utils.AnnotationLabels.GLOBAL_LABELS.values()).index(label_id)
            label_name = list(annotation_utils.AnnotationLabels.GLOBAL_LABELS.keys())[idx]
            overall_dataset_meta_info["total_test_freq"][label_id_updated] = f"{label_name} = {label_value}"
        for label_id, label_value in self.__val_label_cnt.items():
            if self.__label_ids != list(set(annotation_utils.AnnotationLabels.GLOBAL_LABELS.values())):
                # This means that not all classes are selected, but for fasterCNN labels should be [1, ..]
                label_id_updated = self.__label_ids.index(label_id)
            else:
                label_id_updated = label_id

            idx = list(annotation_utils.AnnotationLabels.GLOBAL_LABELS.values()).index(label_id)
            label_name = list(annotation_utils.AnnotationLabels.GLOBAL_LABELS.keys())[idx]
            overall_dataset_meta_info["total_val_freq"][label_id_updated] = f"{label_name} = {label_value}"
        for label_id, label_value in self.__test_raw_label_cnt.items():
            if self.__label_ids != list(set(annotation_utils.AnnotationLabels.GLOBAL_LABELS.values())):
                # This means that not all classes are selected, but for fasterCNN labels should be [1, ..]
                label_id_updated = self.__label_ids.index(label_id)
            else:
                label_id_updated = label_id

            idx = list(annotation_utils.AnnotationLabels.GLOBAL_LABELS.values()).index(label_id)
            label_name = list(annotation_utils.AnnotationLabels.GLOBAL_LABELS.keys())[idx]
            overall_dataset_meta_info["total_test_raw_freq"][label_id_updated] = f"{label_name} = {label_value}"

        overall_dataset_meta_info["volume"] = {
            "img_size": f"{self.__slice_size} x {self.__slice_size} px",
            "test_size": self.__number_images_test,
            "train_size": self.__number_images_train,
            "val_size": self.__number_images_val,

        }
        # TODO: create same dataset meta info file
        with open(os.path.join(self.__dataset_path, "meta.yaml"), "w") as f:
            yaml.dump(dataset_meta_info, f, default_flow_style=False)
            f.write("\n")
            yaml.dump(overall_dataset_meta_info, f, default_flow_style=False)

        return True


    def get_sliced_boxes(self,
                         image_height: int,
                         image_width: int,
                         overlap_height_ratio: float = 0.1,
                         overlap_width_ratio: float = 0.1,
                         slice_height: int = 1024,
                         slice_width: int = 1024,
                         ) -> list[list[int]]:
        """
        This function return list of indexes for slicing image.
        Inspiration from SAHI library
        :param image_height:
        :param image_width:
        :param overlap_height_ratio:
        :param overlap_width_ratio:
        :param slice_height:
        :param slice_width:
        :return:
        """
        slice_bboxes = []
        y_max = y_min = 0

        y_overlap = int(overlap_height_ratio * slice_height)
        x_overlap = int(overlap_width_ratio * slice_width)

        while y_max < image_height:
            x_min = x_max = 0
            y_max = y_min + slice_height
            while x_max < image_width:
                x_max = x_min + slice_width
                if y_max > image_height or x_max > image_width:
                    xmax = min(image_width, x_max)
                    ymax = min(image_height, y_max)
                    xmin = max(0, xmax - slice_width)
                    ymin = max(0, ymax - slice_height)
                    slice_bboxes.append([xmin, ymin, xmax, ymax])
                else:
                    slice_bboxes.append([x_min, y_min, x_max, y_max])
                x_min = x_max - x_overlap
            y_min = y_max - y_overlap
        return slice_bboxes

    def convert_annotation_format(self,
                                  image_height: int,
                                  image_width: int,
                                  annotations: np.array,
                                  ) -> np.array:
        """
        This function convert yolo annotation format to fasterCNN format.
        From relative to absolute px coordinates.
        [class, cx, cy, rw, rh ] -> [class, x1, y1, x2, y2]
        :param image_height:
        :param image_width:
        :param annotations:
        :return:
        """
        ann = np.array(annotations, copy=True, dtype=np.int_)
        x_center_px = annotations[:, 1] * image_width
        y_center_px = annotations[:, 2] * image_height
        width_px = annotations[:, 3] * image_width
        height_px = annotations[:, 4] * image_height
        ann[:, 1] = x_center_px - width_px // 2
        ann[:, 2] = y_center_px - height_px // 2
        ann[:, 3] = x_center_px + width_px // 2
        ann[:, 4] = y_center_px + height_px // 2
        return ann

    def annotation_inside_slice(self, slice_bbox: list[int], annotation: np.array) -> bool:
        """
        Check if annotation is inside bbox more than {overlap-default 70%} threshold.
        :param slice_bbox:
        :param annotation:
        :param overlap:
        :return: True - is inside; False - not inside
        """
        x1, y1, x2, y2 = annotation[1],annotation[2], annotation[3],annotation[4]
        #print(x1, y1, x2, y2, slice_bbox)

        if x2 <= slice_bbox[0]:
            return False
        if y2 <= slice_bbox[1]:
            return False
        if x1 >= slice_bbox[2]:
            return False
        if y1 >= slice_bbox[3]:
            return False
        return True

    def calculate_iou(self, slice_bbox: list[int], annotation: np.array) -> float:
        x1, y1, x2, y2 = annotation[1], annotation[2], annotation[3], annotation[4]
        x_left = max(x1, slice_bbox[0])
        x_right = min(x2, slice_bbox[2])
        y_top = max(y1, slice_bbox[1])
        y_bottom = min(y2, slice_bbox[3])
        assert x_right > x_left, "x_right > x_left"
        assert y_bottom > y_top, "y_bottom > y_top"

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        annotation_area = (x2 - x1) * (y2 -y1)
        return intersection_area / annotation_area

    @staticmethod
    def contains_test_label(label_ids: np.array, test_label_id: int) -> bool:
        exists = np.any(label_ids[:, 0].astype(int) == test_label_id)
        return exists

    @staticmethod
    def remap_labels(annotations: np.array, annotation_data):
        for row in annotations:
            row[0] = annotation_data.label_remap_dict[row[0]]

    @staticmethod
    def update_label_count(label_count_dict: dict, annotations: np.array):
        for a in annotations:
            label_id = a[0]
            label_count_dict[label_id] += 1

    def generate_labels_txt_file(self):
        pass

def parse_opt():
    parser = argparse.ArgumentParser(
        description="Usage: generate dataset for image processing",
        epilog="python generate_dataset.py --label_ids 1,2 --annotation_job_names job1,job2 "
    )

    parser.add_argument('--dataset_name', type=str, help='Name for dataset')
    parser.add_argument('--annotation_job_names', type=str,)
    parser.add_argument('--ssh_private_key_path', type=str,
                        default="/home/hercogs/.ssh/id_rsa_misik01.pub",
                        help="Path to ssh key for access server")
    parser.add_argument('--label_ids', type=str, default=None,
                        help="List of global labels ids to be used")
    parser.add_argument('--overwrite_data', type=bool, default=False,
                        help="Whether to re-download images from server")
    parser.add_argument('--annotation_repo_path', type=str,
                        help="absolute forestai_annotation_repo_path")

    args = parser.parse_args()
    return args

def main(args):

    # Process arguments
    if args.dataset_name:
        dataset_name = args.dataset_name
    else:
        dataset_name = str(datetime.datetime.now())
    annotation_job_names =  args.annotation_job_names.split(',')

    ssh_private_key_path = args.ssh_private_key_path
    label_ids = [int(item) for item in args.label_ids.split(',')]
    overwrite_data = args.overwrite_data
    if args.annotation_repo_path:
        forestai_annotation_repo_path = args.annotation_repo_path
    else:
        forestai_annotation_repo_path = \
            "/home/hercogs/Desktop/Droni/git_repos/faster_cnn/forestai_annotation_data"

    # annotation_job_names = [
    #     "job-22_78520060090_4_4-5-6-7"
    # ]

    am = annotation_utils.AnnotationManager(
        forestai_annotation_repo_path,
        annotation_job_names
    )

    #print(am.list_available_annotations())

    annotation_data_list = am.get_selected_annotations()

    dg = DatasetGenerator(
        ssh_key_path=ssh_private_key_path,
        selected_annotations=annotation_data_list,
        dataset_name = dataset_name,
        use_test_frame = True,
        label_ids = label_ids,  # List
        slice_size = 1024,  # px size of sliced image
        overwrite_data = overwrite_data,
        dataset_format = None,  # Not used yet
    )

    dg.download_images()
    dg.create_dataset_yolov5()


if __name__ == "__main__":
    # python generate_dataset.py --label_ids 1 \
    # --annotation_job_name job-27_80760090047_92_9-10  --dataset_name priede_test
    args = parse_opt()
    main(args)