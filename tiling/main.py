import utils


if __name__ == '__main__':
    # Set basic directories
    src = './images_v3'
    js_save_pth = './images_v3/dataset.json'

    img_paths = utils.get_img_names(src)
    img_names = utils.get_img_names_nopth(src)

    utils.convert_labelme_to_coco(img_paths, js_save_pth)

    row = 4
    col = 6

    dst_tile = './images_v3_tiled'
    out_name = 'dataset_tiled.json'

    utils.tiling_img(row, col, src, dst=dst_tile, names=img_names)
    utils.tiling_bbox(row, col, pth_json=js_save_pth, dst=dst_tile, out_name=out_name)
