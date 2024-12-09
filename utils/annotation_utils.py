import os
import sys, shutil, glob
import logging
import zipfile

import yaml

class AnnotationLabels:
    """
    This class defines global labels for annotation, so all annotation jobs
    use same ids and values. In case of different labels between jobs, this
    class helps to remap them!
    """

    GLOBAL_LABELS = {
        "test_frame":       0,
        "priede":           1,
        "priede_natural":   1,
        "egle":             2,
        "egle_natural":     2,
        "priede_prop":      3,
        "egle_prop":        4,
        "egle_died":        5,
        "priede_died":      6,
        "vaga":             7,
        "pacila":           8,
    }

    TEST_LABEL_NAME =       "test_frame"
    TEST_LABEL_ID =         0

    @staticmethod
    def __len__():
        return len(AnnotationLabels.GLOBAL_LABELS)

    @staticmethod
    def get_remapping(annotation_labels: dict[int, str]) -> dict[int, int]:
        """
        This function creates remap dict, which converts old label id to new one.

        :param annotation_labels: dictionary from annotation task
        :return: remap dictionary
        """
        global_values = AnnotationLabels.GLOBAL_LABELS.keys()
        global_keys = AnnotationLabels.GLOBAL_LABELS.values()

        remap_dict = {}
        for key, value in annotation_labels.items():
            if not value in global_values:
                raise Exception(f"Key '{key}' or value '{value}' not found in global labels")
            remap_dict[key] = AnnotationLabels.GLOBAL_LABELS[value]
        return dict(remap_dict)


class AnnotationData:
    def __init__(self, name, path, yaml_file_path):
        self.name = name            # Annotation name (ex. job-28_44980020033_1_1)
        self._path = path           # Unzipped annotation location for whole job
        self._yaml_file_path = yaml_file_path   # Yaml file path for annotation job
        self.annotations_path = self.get_annotations_path()     # Path where all *txt files are stored
        with open(self._yaml_file_path, "r") as f:
            self.yaml_config = yaml.safe_load(f)

        self.label_dict = self.get_label_dict()
        self.label_remap_dict = AnnotationLabels.get_remapping(self.label_dict)
        self.number_of_classes = len(self.label_dict)

        self.test_label_name = self.yaml_config["label_test_name"]
        if not self.test_label_name == AnnotationLabels.TEST_LABEL_NAME:
            raise Exception(f"Test label name '{self.test_label_name}' in annotation job"\
                            f" {self.name} differs from global test label name"\
                            f" {AnnotationLabels.TEST_LABEL_NAME}")

        self.test_label_id = self.get_label_id(self.test_label_name, self.label_dict)
        self.images_path = None  # Location of unzipped images for this annotation job

        self.kadastrs = self.yaml_config["info"][0]["kadastrs"]
        self.kvartals = self.yaml_config["info"][0]["kvartals"]
        self.nogabals = self.yaml_config["info"][0]["nogabals"]
        self.uav = self.yaml_config["info"][0]["uav"]
        self.datums = self.yaml_config["info"][0]["datums"]

    def get_label_dict(self):
        labels = {}
        with open(os.path.join(self._path, "obj.names"), "r") as f:
            for i, label in enumerate(f.readlines()):
                labels[i] = label.rstrip()
        return labels

    def get_annotations_path(self):
        for root, dirs, files in os.walk(os.path.join(self._path, 'obj_train_data')):
            for file in files:
                return root

    def get_annotated_image_paths(self):
        image_paths = []
        for root, dirs, files in os.walk(os.path.join(self._path, 'obj_train_data')):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.getsize(file_path) > 0:  # we need only non-empty frames
                    image_paths.append(file.replace('.txt', '.JPG'))
        return image_paths

    @staticmethod
    def get_label_id(label_name: str, label_dict: dict) -> int:
        label_id = list(label_dict.keys())[list(label_dict.values()).index(label_name)]
        return label_id


class AnnotationManager:
    """
    This class is responsible for processing annotations from github repo.
    """
    def __init__(self, annotation_repo_path, annotation_job_names: list[str] = None):
        """
        :param annotation_repo_path: location of annotation repo path
        :param annotation_job_names: list of strings with annotation
        """
        self._annotation_data_path = os.path.join(annotation_repo_path, "annotation_data")
        self.__annotation_job_names = annotation_job_names

        self._tmp_file_path = os.path.join(os.getcwd(), "tmp_file_path")
        if not os.path.exists(self._tmp_file_path):
            os.mkdir(self._tmp_file_path)

        self._zip_files = [f for f in glob.glob(self._annotation_data_path + "/*.zip")]
        self._yaml_files = [f for f in glob.glob(self._annotation_data_path + "/*.yaml")]
        self._available_annotations = [os.path.splitext(os.path.basename(f))[0] for f in self._zip_files]

        self._selected_annotations = []  # Storage for selected annotations - AnnotationData object

        if not self.__annotation_job_names:
            self.__annotation_job_names = self._available_annotations
        self.select_annotations(self.__annotation_job_names)

    # def __exit__(self):
    #     if os.path.exists(self._tmp_file_path):
    #         shutil.rmtree(self._tmp_file_path)

    def __del__(self):
        if os.path.exists(self._tmp_file_path):
            shutil.rmtree(self._tmp_file_path)

    def __len__(self):
        """
        :return: number of available annotations
        """
        return len(self._yaml_files)

    def list_available_annotations(self):
        return self._available_annotations

    def select_annotations(self, annotations:list[str] = None):
        if not isinstance(annotations, list):
            user_selected_annotations = input(
                f"Please select annotations to be used from {self._available_annotations}: ").split()
            if not user_selected_annotations:
                raise Exception(f"No annotations chosen. Please rerun and select list of annotations.")
        else:
            user_selected_annotations = annotations

        # Unzip selected annotations
        self.unzip_selected_annotations(user_selected_annotations)

    def unzip_selected_annotations(self, selected_annotations: list):
        """
        This function unzips annotations in self._tmp_file_path
        :return:
        """
        if not isinstance(selected_annotations, list):
            raise Exception

        self._selected_annotations.clear()

        # Check if all annotations exist
        for ann in selected_annotations:
            if ann not in self._available_annotations:
                raise NameError(f"Annotation '{ann}' does not exist. List of" \
                                f" available annotations: {self._available_annotations}")
            # Unzip
            with zipfile.ZipFile(os.path.join(self._annotation_data_path, ann + ".zip"), "r") as zip_ref:
                zip_ref.extractall(path=os.path.join(self._tmp_file_path, ann))

            annotation_data = AnnotationData(
                ann,
                os.path.join(self._tmp_file_path, ann),
                os.path.join(self._annotation_data_path, ann + ".yaml")
            )
            self._selected_annotations.append(annotation_data)

    def get_selected_annotations(self):
        return self._selected_annotations



if __name__ == "__main__":
    pa = "/home/hercogs/Desktop/Droni/git_repos/faster_cnn/forestai_annotation_data"
    ann = ["job-28_44980020033_1_1", "job-17_44980020033_1_1"]
    am = AnnotationManager(pa, ann)

    print(am.list_available_annotations())

