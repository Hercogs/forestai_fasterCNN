import logging, logging.config
import os, shutil
import sys
import zipfile
import tarfile
import datetime
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from utils import server_utils, annotation_utils


class DatasetGenerator:
    def __init__(self, ssh_key_path="/home/hercogs/.ssh/id_rsa_misik01.pub",
                 forestai_annotation_data_path="/home/hercogs/Desktop/Droni/git_repos/faster_cnn/forestai_annotation_data",
                 annotations=None  # list of annotations to be used for dataset
                 ):
        # Set logging
        logging.basicConfig(format="%(filename)s:%(levelname)s - %(message)s", level=logging.ERROR)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        self.server_manager = server_utils.ServerManager(ssh_key_path)
        self.annotation_manager = annotation_utils.AnnotationManager(forestai_annotation_data_path)

        self.selected_annotation_names = self.select_annotations(annotations)
        self.selected_annotations = self.annotation_manager.unzip_selected_annotations(self.selected_annotation_names)

        self.number_images_train = 0
        self.number_images_test = 0
        self.number_images_test_raw = 0

    def select_annotations(self, annotations):
        available_annotations = self.annotation_manager.list_available_annotations()
        if isinstance(annotations, list):
            #TODO: do label mapping if len > 1
            if len(annotations) > 1:
                raise Exception("Label mapping not implemented yet, only 1 annotation can be processed")
            for a in annotations:
                if a not in available_annotations:
                    raise Exception(f"{a} annotation not found in available annotation: {available_annotations}")
            return annotations
        user_selected_annotations = input(
            f"Please select annotations to be used from {available_annotations}: ").split()
        if not user_selected_annotations:
            raise Exception(f"No annotations chosen. Please rerun and select list of annotations.")
        # TODO: do label mapping if len > 1
        if len(user_selected_annotations) > 1:
            raise Exception("Label mapping not implemented yet, only 1 annotation can be processed")
        return user_selected_annotations

    def download_images(self, annotation):
        """
        This function allows to download images from server which are used in annotations.
        :param annotation: annotation object class
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
        self.server_manager.read_from_server(
            annotation.yaml_config["images_locations"],
            os.path.join("datasets", "raw_data_images", annotation.name)
        )
        # Extract archive, if it exists
        # Check for raw data existence as well
        archive_name = os.path.basename(annotation.yaml_config["images_locations"])
        if not os.path.exists(os.path.join("datasets", "raw_data_images", annotation.name, archive_name)) \
                and not os.path.exists(os.path.join("datasets", "raw_data_images", annotation.name)):
            # Extract
            self.logger.info(f"Extracting annotation {annotation.name} images ...")
            with tarfile.open(os.path.join("datasets", "raw_data_images", annotation.name, archive_name),
                              "r") as tar_ref:
                tar_ref.extractall(path=os.path.join("datasets", "raw_data_images", annotation.name))
            # Delete
            self.logger.info(f"Deleting annotation {annotation.name} archive ...")
            os.remove(os.path.join("datasets", "raw_data_images", annotation.name, archive_name))
            # Save path
            annotation.images_path = os.path.join("datasets", "raw_data_images", annotation.name)
        else:
            annotation.images_path = os.path.join("datasets", "raw_data_images", annotation.name)
            self.logger.info(f"Annotation {annotation.name} already downloaded!")

    def create_dataset(self, annotations, dataset_name=None, label_ids=None, use_test_frame=True,
                       slice_size=(1024, 1024), dataset_format=None):
        """
        This functions creates dataset inside ./datasets folder
        :param annotations: list of annotations used in dataset
        :param dataset_name: name of dataset, default value {datetime}
        :param label_ids: list of AnnotationData.GLOBAL_LABELS.keys() of labels ids to be used in dataset
        :param use_test_frame: test data is created if set True
        :param slice_size: pixel size or images in dataset
        :param dataset_format: resnet, yolo, whatever TODO: not implemented
        :return:
        """
        if label_ids is None or len(label_ids) == 0:
            use_all_label_ids = True
        else:
            use_all_label_ids = False
        dataset_name = datetime.datetime.now() if not dataset_name else dataset_name


        # Create dataset output folder structure
        dataset_folder = os.path.join("datasets", dataset_name)
        if os.path.exists(dataset_folder):
            pass
            if dataset_name != "test":
                raise Exception(f"Dataset with name '{dataset_name}' already exists. Delete it or rename")
            else:
                # Delete old dataset
                shutil.rmtree(dataset_folder)
        train_folder = os.path.join(dataset_folder, "train")  # Cropped images
        val_folder = os.path.join(dataset_folder, "val")  # Cropped images
        test_folder = os.path.join(dataset_folder, "test")  # Cropped images
        test_raw_folder = os.path.join(dataset_folder, "test_raw")  # Original images with test_frame label
        os.makedirs(train_folder, exist_ok=True)  # Cropped images
        os.makedirs(val_folder, exist_ok=True)  # Cropped images
        os.makedirs(test_folder, exist_ok=True)  # Cropped images
        os.makedirs(test_raw_folder, exist_ok=True)  # Original images with test_frame label

        #TODO: create *yaml file for dataset

        # Iterate over all annotation objects
        for ann in annotations:
            for file in sorted(os.listdir(ann.annotations_path)):
                # Skip empty annotations
                if not os.path.getsize(os.path.join(ann.annotations_path, file)) > 0:
                    continue
                # Read file content in numpy array
                np_annotations = np.loadtxt(os.path.join(ann.annotations_path, file), dtype=np.float64, ndmin=2)
                #print(file)
                #print(np_annotations)

                is_test_image = self.contains_test_label(np_annotations, ann.test_label_id)
                if is_test_image and not use_test_frame:
                    # Skip annotation file if not use test_frame
                    continue

                self.crop_image(os.path.join(ann.images_path, file).replace(".txt", ".JPG"),
                                np_annotations,
                                dataset_folder,
                                ann.test_label_id,
                                slice_size,
                                label_ids=label_ids,
                                is_test_image=is_test_image)

                # ONly 1 image
                #break
            # Only 1 annotation
            #break

        print(f"In total:\n"
              f"\ttrain images:\t\t",   f"{self.number_images_train}\n"
              f"\ttest images:\t\t",    f"{self.number_images_test}\n"
              f"\ttest_raw images:\t",  f"{self.number_images_test_raw}\n"
        )
        # TODO: create same dataset meta info file

        return True

    def crop_image(self, image_path: str, annotations: np.array, dataset_folder: str,
                   test_label_id: int, slice_size, is_test_image, label_ids=None, dataset_format=None):
        """

        :param image_path:
        :param annotations:
        :param dataset_folder:
        :param test_label_id:
        :param slice_size:
        :param is_test_image: whether it is is test_image or not
        :param label_ids: list of AnnotationData.GLOBAL_LABELS.keys() of labels ids to be used in dataset
        :param dataset_format:
        :return:
        """
        # RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # plt.imshow(RGB_img)
        img = cv.imread(image_path)
        image_height, image_width, _ = img.shape

        sliced_boxes = self.get_sliced_boxes(
            image_height,
            image_width,
            slice_height=slice_size[0],
            slice_width=slice_size[1]
        )

        annotations_in_pxs = self.convert_annotation_format(
            image_height,
            image_width,
            annotations
        )

        train_folder = os.path.join(dataset_folder, "train")  # Cropped images
        val_folder = os.path.join(dataset_folder, "val")  # Cropped images
        test_folder = os.path.join(dataset_folder, "test")  # Cropped images
        test_raw_folder = os.path.join(dataset_folder, "test_raw")  # Original images with test_frame label

        # Lamda function for checking label id
        label_in_use = lambda label_id: np.array([True for _ in label_id]) if label_ids is None else np.array([i in label_ids for i in label_id])

        if is_test_image:
            print(f"Testa image: {image_path}")
            # Save full image for later tets purpose
            self.number_images_test_raw += 1
            shutil.copy(image_path, os.path.join(test_raw_folder, str(self.number_images_test_raw) + ".JPG"))
            # Delete test label TODO: remove mext line
            ann = np.delete(annotations_in_pxs, np.where(annotations_in_pxs[:, 0] == test_label_id), axis=0)
            # Delete other labels
            ann = np.delete(ann, np.where(np.invert(label_in_use(ann[:, 0]))), axis=0)
            np.savetxt(
                fname=os.path.join(test_raw_folder, str(self.number_images_test_raw) + ".txt"),
                X=ann.astype(int),
                fmt="%s",
                delimiter=","
            )

        self.logger.info(f"Processsing: {image_path}")

        for sliced_box in sliced_boxes:
            # Colect all anotion
            sliced_annotation_list: list = []  # List to keep annotations for specific slice
            for annotation_in_pxs in annotations_in_pxs:
                is_in = self.annotation_inside_slice(sliced_box, annotation_in_pxs)
                if not is_in:
                    continue
                iou = self.calculate_iou(sliced_box, annotation_in_pxs)
                if not (iou > 0.7):
                    continue
                if label_in_use(np.array([annotation_in_pxs[0]])):
                    sliced_annotation_list.append(annotation_in_pxs)

            # print(sliced_annotation_list)
            # print(sliced_box)
            # print("-"*10)

            # Convert global annotation frame to local frame
            for i, sliced_annotation in enumerate(sliced_annotation_list):
                sliced_annotation_list[i] = [
                    sliced_annotation[0],
                    sliced_annotation[1] - sliced_box[0],
                    sliced_annotation[2] - sliced_box[1],
                    sliced_annotation[3] - sliced_box[0],
                    sliced_annotation[4] - sliced_box[1],
                ]

            crop = img[sliced_box[1]:sliced_box[3], sliced_box[0]:sliced_box[2]]

            if is_test_image:
                self.number_images_test += 1
                # Save cropped image
                cv.imwrite(os.path.join(test_folder, str(self.number_images_test) + ".JPG"), crop)
                # Save *txt annotation file
                np.savetxt(
                    fname=os.path.join(test_folder, str(self.number_images_test) + ".txt"),
                    X=np.array(sliced_annotation_list).astype(int),
                    fmt="%s",
                    delimiter=","
                )
            else:
                self.number_images_train += 1
                # Save cropped image
                cv.imwrite(os.path.join(train_folder, str(self.number_images_train) + ".JPG"), crop)
                # Save *txt annotation file
                np.savetxt(
                    fname=os.path.join(train_folder, str(self.number_images_train) + ".txt"),
                    X=np.array(sliced_annotation_list).astype(int),
                    fmt="%s",
                    delimiter=","
                )

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

    def generate_labels_txt_file(self):
        pass


if __name__ == "__main__":
    dg = DatasetGenerator(annotations=["job-27_80760090047_92_9-10"])
    # Use dg.available_annotations to see all availbale annotation
    # Use dg.select_annotations(["ann1", "ann2", ...]) to choose annotation

    dg.logger.info(f"User has selected {dg.selected_annotation_names} annotations.")

    # Step 1 - download all images for annotations
    for a in dg.selected_annotations:
        dg.download_images(a)

    status = dg.create_dataset(
        annotations=dg.selected_annotations,
        dataset_name="priede",
        use_test_frame=True,
        label_ids=[1] # keys from label dict; 1 - priede
    )
    if status:
        dg.logger.info(f"Dataset generated successfully")
    else:
        dg.logger.info(f"Dataset generation failed")
        sys.exit(0)
