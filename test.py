

class AnnotationLabels:
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

    @staticmethod
    def __len__():
        return len(AnnotationLabels.GLOBAL_LABELS)

    @staticmethod
    def get_remapping(annotation_labels: dict) -> dict:
        global_values = AnnotationLabels.GLOBAL_LABELS.keys()
        global_keys = AnnotationLabels.GLOBAL_LABELS.values()

        remap_dict = {}
        for key, value in annotation_labels.items():
            if not key in global_keys or not value in global_values:
                raise Exception(f"Key '{key}' or value '{value}' not found in global labels")
            remap_dict[key] = AnnotationLabels.GLOBAL_LABELS[value]
        return dict(remap_dict)



if __name__ == "__main__":
    test_dict = {
        0: "egle",
        1: "priede",
        2: "egle_died",
        3: "priede_natural",
        4: "vaga",
        5: "test_frame",
        6: "egle_prop",
        7: "priede_prop"
    }

    al = AnnotationLabels()
    print(len(al))

    remap = al.get_remapping(test_dict)

    print(remap)
