import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np


def main(dataset_name, number_of_images):
    dataset_folder = os.path.join("datasets", dataset_name)
    dataset_train = os.path.join(dataset_folder, "train")
    i = 0

    print(dataset_train)
    for root, dirs, files in os.walk(dataset_train):
        files.sort(key=lambda f: int(f[:-4]))
        for f in files:
            file_path = os.path.join(root, f)
            if not f.endswith(".txt"):
                continue
            if os.path.getsize(file_path) == 0:
                continue
            image_path = file_path.replace(".txt", ".JPG")
            print(image_path)
            img = cv.imread(image_path)
            ann = np.loadtxt(file_path, dtype=int, delimiter=",", ndmin=2)
            for a in ann:
                img = cv.rectangle(img, (a[1], a[2]), (a[3], a[4]), (0, 255, 0), 5)

            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            #
            plt.imshow(img)
            plt.show()
            i += 1
            if i == number_of_images:
                break



if __name__ == "__main__":
    dataset = "2024-09-18 11:32:03.928673"
    main(dataset, 3)