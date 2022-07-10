import os
import argparse
import json

import numpy as np
import glob
import PIL.Image
import base64
import io
import numpy as np
import PIL.Image
import PIL.ImageDraw
from numpyencoder import NumpyEncoder


def img_b64_to_arr(img_b64):
    f = io.BytesIO()
    f.write(base64.b64decode(img_b64))
    img_arr = np.array(PIL.Image.open(f))
    return img_arr

class labelme2coco(object):
    def __init__(self, labelme_json=[], save_json_path="./coco.json"):
        """
        :param labelme_json: the list of all labelme json file paths
        :param save_json_path: the path to save new json
        """
        self.labelme_json = labelme_json
        self.save_json_path = save_json_path
        self.images = []
        self.categories = []
        self.annotations = []
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0

        self.save_json()


    def data_transfer(self):
        for num, json_file in enumerate(self.labelme_json):
            with open(json_file, "r") as fp:
                data = json.load(fp)
                self.images.append(self.image(data, num))
                for shapes in data["shapes"]:
                    label = shapes["label"].split("_")
                    if label not in self.label:
                        self.label.append(label)
                    points = shapes["points"]
                    self.annotations.append(self.annotation(points, label, num))
                    self.annID += 1

        # Sort all text labels so they are in the same order across data splits.
        self.label.sort()
        for label in self.label:
            self.categories.append(self.category(label))
        for annotation in self.annotations:
            annotation["category_id"] = self.getcatid(annotation["category_id"])

    def image(self, data, num):
        image = {}
        # img = img_b64_to_arr(data["imageData"])
        height, width = data["imageHeight"], data["imageWidth"]

        image["height"] = height
        image["width"] = width
        image["id"] = num
        image["file_name"] = data["imagePath"].split("/")[-1]

        self.height = height
        self.width = width

        return image

    def category(self, label):
        category = {}
        category["supercategory"] = label[0]
        category["id"] = len(self.categories)
        category["name"] = label[0]
        return category

    def annotation(self, points, label, num):
        annotation = {}

        width = points[1][0] - points[0][0]
        height = points[1][1] - points[0][1]

        annotation["segmentation"] = []
        annotation["iscrowd"] = 0
        annotation["area"] = width*height
        annotation["image_id"] = num

        annotation["bbox"] = list(map(float, self.getbbox(points)))

        annotation["category_id"] = label[0]  # self.getcatid(label)
        annotation["id"] = self.annID
        return annotation

    def getcatid(self, label):
        for category in self.categories:
            if label == category["name"]:
                return category["id"]
        print("label: {} not in categories: {}.".format(label, self.categories))
        exit()
        return -1

    def getbbox(self, points):
        polygons = points
        mask = self.polygons_to_mask([self.height, self.width], polygons)
        return self.mask2box(mask)

    def mask2box(self, mask):

        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]

        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        return [
            left_top_c,
            left_top_r,
            right_bottom_c - left_top_c,
            right_bottom_r - left_top_r,
        ]

    def polygons_to_mask(self, img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask

    def data2coco(self):
        data_coco = {}
        data_coco["images"] = self.images
        data_coco["categories"] = self.categories
        data_coco["annotations"] = self.annotations
        return data_coco

    def save_json(self):
        print(f"Strat saving COCO json file for given file list to {self.save_json_path}")
        self.data_transfer()
        self.data_coco = self.data2coco()

        os.makedirs(
            os.path.dirname(os.path.abspath(self.save_json_path)), exist_ok=True
        )

        json.dump(self.data_coco, open(self.save_json_path, "w"), cls= NumpyEncoder, indent=4)
        print(f"COCO json file is created as {self.save_json_path}.")