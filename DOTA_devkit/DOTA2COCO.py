import dota_utils as util
import os
import cv2
import json
from PIL import Image
import tqdm

wordname_15 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship',
               'tennis-court',
               'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool',
               'helicopter']

wordname_16 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship',
               'tennis-court',
               'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool',
               'helicopter', 'container-crane']

wordname_text = ['text']


def DOTA2COCOTrain(srcpath, destfile, class_map: dict, class_names, difficult='2', ext='.png'):
    # set difficult to filter '2', '1', or do not filter, set '-1'

    imageparent = os.path.join(srcpath, 'images')
    labelparent = os.path.join(srcpath, 'labelTxt')

    data_dict = {}
    data_dict['images'] = []
    data_dict['categories'] = []
    data_dict['annotations'] = []
    used = set()
    for sub_cat_name, cat_name in class_map.items():
        cat_id = class_names.index(cat_name) + 1
        if cat_id not in used:
            single_cat = {'id': cat_id, 'name': cat_name, 'supercategory': cat_name}
            data_dict['categories'].append(single_cat)
        used.add(cat_id)

    inst_count = 1
    image_id = 1
    with open(destfile, 'w') as f_out:
        filenames = util.GetFileFromThisRootDir(labelparent)
        for file in tqdm.tqdm(filenames):
            basename = util.custombasename(file)
            # image_id = int(basename[1:])

            imagepath = os.path.join(imageparent, basename + ext)
            img = Image.open(imagepath)
            height = img.height
            width = img.width

            single_image = {}
            single_image['file_name'] = basename + ext
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)

            # annotations
            objects = util.pars_fair1_json2(file)

            show_data = True
            if show_data:
                import numpy as np
                import random
                img = cv2.imread(imagepath)
                height, width, c = img.shape
                for obj in objects:
                    pts = np.array(obj['poly'], np.int32).reshape((4, 2))
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(img, [pts], True, [random.randint(0, 255) for _ in range(2)], 2)

                cv2.imshow("img", img)
                if cv2.waitKey(0) == 27:
                    exit()
            for obj in objects:
                if obj['difficult'] == difficult:
                    print('difficult: ', difficult)
                    continue
                single_obj = {}
                # single_obj['area'] = obj['area']
                if obj['name'].lower() not in class_map:
                    print(obj['name'])
                class_name = class_map[obj['name'].lower()]
                single_obj['category_id'] = class_names.index(class_name) + 1
                single_obj['segmentation'] = []
                single_obj['segmentation'].append(obj['poly'])
                single_obj['iscrowd'] = 0
                xmin, ymin, xmax, ymax = min(obj['poly'][0::2]), min(obj['poly'][1::2]), \
                                         max(obj['poly'][0::2]), max(obj['poly'][1::2])

                width, height = xmax - xmin, ymax - ymin
                single_obj['bbox'] = xmin, ymin, width, height
                # modified
                single_obj['area'] = width * height
                single_obj['image_id'] = image_id
                data_dict['annotations'].append(single_obj)
                single_obj['id'] = inst_count
                inst_count = inst_count + 1
            image_id = image_id + 1
        json.dump(data_dict, f_out)


def DOTA2COCOTest(srcpath, destfile, class_map, class_names, ext='.png'):
    imageparent = os.path.join(srcpath, 'images')
    data_dict = {}

    data_dict['images'] = []
    data_dict['categories'] = []
    for sub_cat_name, cat_name in class_map.items():
        single_cat = {'id': class_names.index(cat_name) + 1, 'name': cat_name, 'supercategory': cat_name}
        data_dict['categories'].append(single_cat)

    image_id = 1
    with open(destfile, 'w') as f_out:
        filenames = util.GetFileFromThisRootDir(imageparent)
        for file in tqdm.tqdm(filenames):
            basename = util.custombasename(file)
            imagepath = os.path.join(imageparent, basename + ext)
            img = Image.open(imagepath)
            height = img.height
            width = img.width

            single_image = {}
            single_image['file_name'] = basename + ext
            single_image['id'] = image_id
            single_image['width'] = width
            single_image['height'] = height
            data_dict['images'].append(single_image)

            image_id = image_id + 1
        json.dump(data_dict, f_out)


if __name__ == '__main__':
    # DOTA2COCOTrain(r'/home/dj/code/mmdetection_DOTA/data/dota1_1024_v2/trainval1024',
    #                r'/home/dj/code/mmdetection_DOTA/data/dota1_1024_v2/trainval1024/DOTA_trainval1024.json',
    #                wordname_15)
    # DOTA2COCOTrain(r'/home/dj/code/mmdetection_DOTA/data/dota1_1024_v2/trainval1024_ms',
    #                r'/home/dj/code/mmdetection_DOTA/data/dota1_1024_v2/trainval1024_ms/DOTA_trainval1024_ms.json',
    #                wordname_15)
    # DOTA2COCOTest(r'/workfs/jmhan/dota15_1024_ms/test1024',
    #               r'/workfs/jmhan/dota15_1024_ms/test1024/DOTA_test1024.json',
    #               wordname_16)
    # DOTA2COCOTest(r'/workfs/jmhan/dota15_1024_ms/test1024_ms',
    #               r'/workfs/jmhan/dota15_1024_ms/test1024_ms/DOTA_test1024_ms.json',
    #               wordname_16)
    # DOTA2COCOTrain(r'data/MSRA500_DOTA/train',
    #                r'data/MSRA500_DOTA/train/train.json',
    #                wordname_text, ext='.JPG')

    # DOTA2COCOTest(r'data/MSRA500_DOTA/test',
    #               r'data/MSRA500_DOTA/test/test.json',
    #               wordname_text, ext='.JPG')

    # DOTA2COCOTrain(r'data/RCTW/train',
    #                r'data/RCTW/train/train.json',
    #                wordname_text, ext='.jpg')
    #
    # DOTA2COCOTest(r'data/RCTW/test',
    #                r'data/RCTW/test/test.json',
    #                wordname_text, ext='.jpg')

    DOTA2COCOTrain(r'/workfs/jmhan/dota_ms_1024/trainval_split',
                   r'/workfs/jmhan/dota_ms_1024/trainval_split/DOTA_trainval1024_ms.json',
                   wordname_15)
    DOTA2COCOTest(r'/workfs/jmhan/dota_ms_1024/test_split',
                  r'/workfs/jmhan/dota_ms_1024/test_split/DOTA_test1024_ms.json',
                  wordname_15)
