import json

with open("./images_v3_tiled/dataset_tiled.json", "r") as f:
    data = json.load(f)

img_lst = []
for i in data["annotations"]:
    img_id = i["image_id"]
    if img_id not in img_lst:
        img_lst.append(img_id)

print(sorted(img_lst))
print(len(img_lst))