import os
import shutil
import json
from labelme_to_coco import *
import random

from PIL import Image
from pycocotools.coco import COCO
from collections import OrderedDict


def get_img_names(src):  # Get list of (image) file names with extension
    file_list = [os.path.join(src, f) for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]  # Get a list of names of files in the folder
    file_list = [i.split('.') for i in file_list]
    file_list = [i for i in file_list if i[-1] != 'json']
    file_list = ['.'.join(i) for i in file_list]
    return file_list

def get_img_names_nopth(src):
    file_list = [os.path.join(src, f) for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]  # Get a list of names of files in the folder
    file_list = [i.split('.') for i in file_list]
    file_list = [i for i in file_list if i[-1] != 'json']
    file_list = ['.'.join(i) for i in file_list]
    file_list = [i.split('\\')[-1] for i in file_list]
    return file_list

def get_js_names(pth):  # Get list of (json) file names with extension
    file_list = os.listdir(pth)  # Get a list of names of files in the folder
    file_list = [i.split('.') for i in file_list]
    file_list = [i for i in file_list if i[-1] == 'json']
    file_list = ['.'.join(i) for i in file_list]
    return file_list

def get_names_without_ext(pth):
    file_list = os.listdir(pth)  # Get a list of names of files in the folder
    file_list = [i.split('.') for i in file_list]
    file_list = [i for i in file_list if i[1] != 'json']
    file_list = [i[0] for i in file_list]

    return file_list


def cp_imgs_with_names(names, src, dst):
    for name in names:
        src_pth = os.path.join(src, name)
        dst_pth = os.path.join(dst, name)

        if os.path.exists(dst):
            shutil.rmtree(dst)

        os.makedirs(os.path.dirname(dst), exist_ok=True)

        shutil.copyfile(src_pth, dst_pth)
    return None

def ext_to_js(file_name_lst):
    return ['.'.join(i.split('.')[:-1]) + '.json' for i in file_name_lst]

def tiling_img(row, col, src, dst, names):  # Tiling images into n slices in new folder
    # Copy the folder with original (whole) images
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    # Remove annotation files
    temp = os.listdir(dst)
    for item in temp:
        if item.endswith(".json"):
            os.remove(os.path.join(dst, item))

    # Slice each original image into n slices and remove original one
    for i in names:
        file_pth = os.path.join(dst, i)
        image = Image.open(file_pth)
        image_size = image.size

        w = image_size[0] // col
        h = image_size[1] // row

        for a in range(0, row):
            for b in range(0, col):

                image_sliced = image.crop((w*b, h*a, w*b + w, h*a + h))
                extension = i.split('.')[-1]
                name = i[:-(len(extension) + 1)]
                slice_pth = os.path.join(dst, f'{name}_0{a+1}_0{b+1}.{extension}')
                image_sliced.save(slice_pth)
        os.remove(file_pth)
    print(f'{len(names)} images are sliced! (Result in {len(os.listdir(dst))} slices)')
    return None


def tiling_bbox(row, col, pth_json, dst, out_name):
    coco = COCO(pth_json)  # Load COCO annotation

    new_json = OrderedDict()
    new_json['images'] = []
    new_json['annotations'] = []
    new_json['categories'] = coco.loadCats(0)  # Assume there is only one category

    img_id = 0  # Image ID
    ann_id = 1  # Annotation ID
    annot_loss = 0

    # Iterate over original (whole) images
    for i in coco.imgs:

        # New height and width for sliced image
        h = coco.loadImgs(i)[0]['height'] // row
        w = coco.loadImgs(i)[0]['width'] // col

        # Get image name without extension & extension respectively
        img_name = '.'.join(coco.loadImgs(i)[0]['file_name'].split('.')[:-1])
        ext = coco.loadImgs(i)[0]['file_name'].split('.')[-1]

        # Give names new sliced images in form of: IMG_NAME_0A_0B.ext
        # Set fields 'id', 'height', 'width', and 'file_name'
        for j in range(1, col + 1):
            for k in range(1, row + 1):
                slice_name = img_name + f'_0{j}_0{k}.{ext}'
                img_id += 1
                dct_img_temp = {"height": h, "width": w, "id": img_id, "file_name": slice_name}
                new_json['images'].append(dct_img_temp)

        # Get all anotations (a list with annotations in corresponding image)
        lst_annots = coco.getAnnIds(i)

        for j in lst_annots:
            annot = coco.loadAnns(j)[0]
            category_id = annot["category_id"]
            x1, y1 = annot["bbox"][0], annot["bbox"][1]

            # COCO bounding box format is [top left x position, top left y position, width, height].
            bbox_w, bbox_h = annot["bbox"][2], annot["bbox"][3]

            # where a & b mean: filename_0a_0b.ext
            slice_num_a = x1 // w
            slice_num_b = y1 // h
            new_x1, new_y1 = x1 % w, y1 % h

            new_bbox = [new_x1, new_y1, bbox_w, bbox_h]
            area = bbox_w * bbox_h
            img_id = int(row * col * (i) + slice_num_a + slice_num_b * row + 1)

            if new_x1 + bbox_w > w or new_y1 + bbox_h > h:
                annot_loss += 1
            else:
                new_annot = {"iscrowd": 0, "image_id": img_id,
                             "bbox": new_bbox, "segmentation": [],
                             "category_id": category_id, "id": ann_id, "area": area}
                new_json["annotations"].append(new_annot)
                ann_id += 1

    # print(json.dumps(new_json, ensure_ascii=False, indent=4))  # Print JSON file on memory
    with open(dst + f'/{out_name}', 'w') as outfile:
        json.dump(new_json, outfile, indent=4)
    print(f'{annot_loss} of annotations has been lost.')


def get_coco_img_lst(pth_json):
    with open(pth_json, "r") as data_json:
        data_python = json.load(data_json)
    imgs = data_python["images"]
    img_lst = list()
    for i in imgs:
        img_lst.append(i["file_name"])
    return img_lst


def img_dir_by_type(src, img_lst, img_type, data_type):
    for i in img_type:

        dst = os.path.join(src, f'{i}_{data_type}')
        json_lst = ['.'.join(i.split('.')[:-1]) + '.json' for i in img_lst]

        if os.path.isdir(dst):
            shutil.rmtree(dst)
        else:
            os.mkdir(f'{i}_{data_type}')

        for j in img_type[i]:
            img_lst = [k for k in img_lst if j in i]
            for img in img_lst:
                shutil.copy(os.path.join(src, img), os.path.join(dst, img))
            for js in json_lst:
                shutil.copy(os.path.join(src, js), os.path.join(dst, img))

        labelme2coco.convert(dst, dst, 1)    # dataset.json

    return None



def random_split(src, train_dir, val_dir, ratio = 0.7):

    img_all_lst = get_img_names(src)
    img_num = len(img_all_lst)

    train_num = int(img_num * ratio)

    random.shuffle(img_all_lst)

    img_train_lst = img_all_lst[:train_num]
    img_val_lst = img_all_lst[train_num:]

    return img_train_lst, img_val_lst


def type_split(img_lst, type):
    img_dct = dict()
    for i in type:
        img_dct[i] = [img for img in img_lst if i in img]
    return img_dct


def type_count(img_lst, type):
    img_dct = dict()
    for i in type:
        img_dct[i] = len([img for img in img_lst if i in img])
    return img_dct


def convert_labelme_to_coco(img_lst, save_pth):
    js_lst = ext_to_js(img_lst)
    labelme2coco(js_lst, save_pth)
    return None


