import utils
import parsing

if __name__ == '__main__':
    # Set basic directories
    src = './images_v3'
    js_save_name = 'dataset.json'
    js_save_pth = os.path.join(src, js_save_name)

    img_paths = utils.get_img_names(src)
    img_names = utils.get_img_names_nopth(src)

    utils.convert_labelme_to_coco(img_paths, js_save_pth)

    row = 4
    col = 6

    dst_tile = './images_v3_tiled'
    out_name = 'dataset_tiled.json'
    js_tiled_pth = os.join.path(dst_tile, out_name)

    utils.tiling_img(row, col, src, dst=dst_tile, names=img_names)
    utils.tiling_bbox(row, col, pth_json=js_save_pth, dst=dst_tile, out_name=out_name)

    pos_img_names = parsing.get_pos_neg_names(js_tiled_pth, pos=True)
    neg_img_names = parsing.get_pos_neg_names(js_tiled_pth, pos=False)

    image_folder_pth = './iamges_v3_classification'

    dst_tile_pos = os.path.join(dst_tile, 'positive')
    dst_tile_neg = os.path.join(dst_tile, 'negative')

    utils.cp_imgs_with_names(pos_img_names, dst_tile, dst_tile_pos)
    utils.cp_imgs_with_names(neg_img_names, dst_tile, dst_tile_neg)
