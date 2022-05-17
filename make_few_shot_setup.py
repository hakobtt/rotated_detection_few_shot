import argparse
import json
import math
import os
import random

import numpy as np
from pycocotools.coco import COCO
import cv2


def load_annotations(ann_file, data_root):
    """Load annotation from COCO style annotation file.

    Args:
        ann_file (str): Path of annotation file.

    Returns:
        list[dict]: Annotation info from COCO api.
    """

    coco = COCO(ann_file)
    cats = coco.cats
    print(cats)

    list_hws = []

    img_ids = coco.getImgIds()
    cat_cnt = {}
    for cat in coco.cats.values():
        cat_cnt[cat['id']] = 0
    for img_id in img_ids[::]:
        info = coco.loadImgs([img_id])[0]
        img_path = os.path.join(data_root, info['file_name'])
        print(img_path)
        ann_ids = coco.getAnnIds([img_id])
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            cat_cnt[ann['category_id']] += 1
            bbox = ann['bbox']
            x, y = int(bbox[0]), int(bbox[1])
            w, h = int(bbox[2]), int(bbox[3])
            list_hws.append(w / h)

        show = True
        if show:
            img = cv2.imread(img_path)
            for ann in anns:
                bbox = ann['bbox']

                x, y = int(bbox[0]), int(bbox[1])
                w, h = int(bbox[2]), int(bbox[3])
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                text = str(cats[ann['category_id']]['name'])
                cv2.putText(img, text, (x, y),
                            cv2.QT_FONT_NORMAL, 0.5, (128, 128, 255))
            print(info)
            cv2.imshow('img', img)
            if cv2.waitKey(0) == 27:
                exit()

    print(cat_cnt)
    z = np.array(list_hws).reshape((-1, 1))
    z = np.float32(z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # Set flags (Just to avoid line break in the code)
    flags = cv2.KMEANS_RANDOM_CENTERS
    # Apply KMeans
    compactness, labels, centers = cv2.kmeans(z, 10, None, criteria, 10, flags)
    print(sorted(centers.reshape((10,))))


def create_few_shot(ann_file, data_root_dir, each_class_cnt):
    """Load annotation from COCO style annotation file.

    Args:
        ann_file (str): Path of annotation file.

    Returns:
        list[dict]: Annotation info from COCO api.
    """
    with open(ann_file) as f:
        annotations_dict = json.load(f)
    # print(annotations_dict['annotations'])

    coco = COCO(ann_file)
    print(coco.cats)
    cat_to_cnt = dict([(c, 0) for c in coco.cats.keys()])
    new_annotations = []
    new_images = []
    img_ids = coco.getImgIds()
    confusion_mat = dict([(c, dict((c1,0) for c1 in coco.cats.keys())) for c in coco.cats.keys()])
    for img_id in img_ids[:]:
        if min(cat_to_cnt.values()) >= each_class_cnt:
            break
        ann_ids = coco.getAnnIds([img_id])
        anns = coco.loadAnns(ann_ids)
        if len(anns) == 0:
            continue
        min_cnt = min([cat_to_cnt[ann["category_id"]] for ann in anns])

        if min_cnt < each_class_cnt:
            new_annotations += anns
            new_images.append(img_id)
            for ann in anns:
                if cat_to_cnt[ann["category_id"]] == min_cnt:
                    curr_class_cnts = confusion_mat[ann["category_id"]]
                    for ann1 in anns:
                        curr_class_cnts[ann1["category_id"]]+=1
                    cat_to_cnt[ann["category_id"]] += 1
                    break


    print("         ", [c["name"] for c in coco.cats.values()])
    for cc, vals in confusion_mat.items():

        print(coco.cats[cc]["name"], *tuple(vals.values()))
    print(confusion_mat)
    annotations_dict['images'] = coco.loadImgs(new_images)
    annotations_dict['annotations'] = new_annotations
    with open(os.path.join(data_root_dir, f"../few_shot_{each_class_cnt}.json"), 'w') as f:
        json.dump(annotations_dict, f)


def main():
    data_root_dir = "/mnt/data/datasets/gaofen/FAIR1M2.0/fair1_1000/val1000/images"
    annotation_file = "/mnt/data/datasets/gaofen/FAIR1M2.0/fair1_1000/val1000/val1000_5classes.json"
    data_root_dir = "/mnt/data/datasets/gaofen/FAIR1M2.0/fair1_1000/train1000/images"
    annotation_file = "/mnt/data/datasets/gaofen/FAIR1M2.0/fair1_1000/train1000/train1000_5classes.json"

    create_few_shot(annotation_file,
                    data_root_dir, 50)
    load = True
    if load:
        load_annotations(annotation_file, data_root_dir)


if __name__ == '__main__':
    main()
