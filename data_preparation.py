import os
import csv


class DataPreparation:
    def __init__(self, relative_path: str = "/DataSets/Drones/") -> None:
        self.current_path = os.getcwd()
        self.base_path = self.current_path + relative_path

    def _get_filenames(self, directory: str, extension: str) -> list:
        return [
            self.base_path + directory + filename for filename in os.listdir(self.base_path + directory)
            if filename.split(".")[1:] == extension.split(".")
        ]

    @staticmethod
    def get_annotations_data(image_filenames: list, box_filenames: list) -> list:
        annotations_data = []
        for index, box_filename in enumerate(box_filenames):
            with open(box_filename, "r") as box_file:
                box_data = box_file.read()
                rows = box_data.split("\n")
                for i in range(0, len(rows) - 1):
                    annotations_data.append(
                        [image_filenames[index]] + [int(x) for x in rows[i].split("\t")] + ['drone']
                    )
        return annotations_data

    def save_csv_file(self, filename: str, data: list) -> None:
        with open(self.current_path + "/" + filename, 'w', newline='') as csv_file:
            wr = csv.writer(csv_file)
            wr.writerows(data)

    def get_positive_data(self,
                          directory: str = "positive/",
                          image_extension: str = "jpg",
                          box_extension: str = "bboxes.tsv") -> tuple:
        image_filenames = self._get_filenames(directory, image_extension)
        box_filenames = self._get_filenames(directory, box_extension)
        return image_filenames, box_filenames

    def get_test_data(self,
                      directory: str = "testImages/",
                      image_extension: str = "jpg",
                      box_extension: str = "bboxes.tsv") -> tuple:
        image_filenames = self._get_filenames(directory, image_extension)
        box_filenames = self._get_filenames(directory, box_extension)
        return image_filenames, box_filenames

    def get_negative_data(self,
                          directory: str = "negative/",
                          image_extension: str = "jpg") -> tuple:
        image_filenames = self._get_filenames(directory, image_extension)
        return image_filenames, []

if __name__ == "__main__":
    dp = DataPreparation()
    positive_image_filenames, positive_box_filenames = dp.get_positive_data()
    negative_image_filenames, negative_box_filenames = dp.get_negative_data()
    test_image_filenames, test_box_filenames = dp.get_test_data()

    positive_annotated_data = dp.get_annotations_data(positive_image_filenames, positive_box_filenames)
    negative_annotated_data = [
        [negative_image_filenames[i], "", "", "", "", ""] for i in range(0, len(negative_image_filenames))
    ]

    train_data = positive_annotated_data + negative_annotated_data
    test_data = dp.get_annotations_data(test_image_filenames, test_box_filenames)

    labels_data = [["drone", 0], ["dummy", 1]]
    dp.save_csv_file("annotations.csv", train_data)
    dp.save_csv_file("validation_annotations.csv", test_data)
    dp.save_csv_file("classes.csv", labels_data)
