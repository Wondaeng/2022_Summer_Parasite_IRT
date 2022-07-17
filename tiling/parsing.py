import json

def get_pos_neg_names(coco_js_pth, pos=True):  # Get names of (positive or negative) images with bounding box(es)
    with open(coco_js_pth, "r") as f:
        data = json.load(f)

    # Parse ID of images with at least 1 annotation
    img_id_lst = []
    for i in data["annotations"]:
        img_id = i["image_id"]
        if img_id not in img_id_lst:
            img_id_lst.append(img_id)

    # Parse names of according images of the list of IDs
    img_name_lst = []
    for i in data["images"]:
        if pos:  # If we need positive images
            if i["id"] in img_id_lst:
                img_name_lst.append(i["file_name"])
        else:  # If we need negative images
            if i["id"] not in img_id_lst:
                img_name_lst.append(i["file_name"])
    return img_name_lst


