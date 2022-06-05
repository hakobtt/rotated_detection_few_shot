import argparse
import json
import math
import os
import random

import numpy as np

from mmdet_custom.core import mask2poly
from pycocotools.coco import COCO
import cv2
import pycocotools.mask as maskUtils
import tqdm


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
                pts = np.array(ann['segmentation'], np.int32).reshape((4, 2))
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(img, [pts], True, [random.randint(0, 255) for _ in range(2)], 2)

                x, y = int(bbox[0]), int(bbox[1])
                w, h = int(bbox[2]), int(bbox[3])
                # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                text = str(cats[ann['category_id']]['name'])
                cv2.putText(img, text, (x, y),
                            cv2.QT_FONT_NORMAL, 0.5, (128, 128, 255))
            cv2.imshow('img', img)
            if cv2.waitKey(0) == 27:
                exit()
    print(f"{'category name' :<20} | {'count':<10}")
    for cat_id, cat_cnt in cat_cnt.items():
        print(f"{cats[cat_id]['name'] :<20} | {cat_cnt:<10}")

    z = np.array(list_hws).reshape((-1, 1))
    z = np.float32(z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # Set flags (Just to avoid line break in the code)
    flags = cv2.KMEANS_RANDOM_CENTERS
    # Apply KMeans
    compactness, labels, centers = cv2.kmeans(z, 10, None, criteria, 10, flags)
    # print(sorted(centers.reshape((10,))))


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
    annotation_file = "/mnt/data/datasets/gaofen/FAIR1M2.0/fair1_1000/val1000/val1000.json"
    # coco_find_grin = COCO(annotation_file)
    print(coco.cats)
    cat_to_cnt = dict([(c, 0) for c in coco.cats.keys()])
    new_annotations = []
    new_images = []
    img_ids = coco.getImgIds()
    confusion_mat = dict([(c, dict((c1, 0) for c1 in coco.cats.keys())) for c in coco.cats.keys()])
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
            # new_annotations += coco_find_grin.loadAnns(coco_find_grin.getAnnIds([img_id]))
            new_images.append(img_id)
            for ann in anns:
                if cat_to_cnt[ann["category_id"]] == min_cnt:
                    curr_class_cnts = confusion_mat[ann["category_id"]]
                    for ann1 in anns:
                        curr_class_cnts[ann1["category_id"]] += 1
                    cat_to_cnt[ann["category_id"]] += 1
                    break

    print("         ", [c["name"] for c in coco.cats.values()])
    for cc, vals in confusion_mat.items():
        print(coco.cats[cc]["name"], *tuple(vals.values()))
    print(confusion_mat)

    # annotations_dict['categories'] = coco_find_grin.dataset['categories']
    annotations_dict['images'] = coco.loadImgs(new_images)
    annotations_dict['annotations'] = new_annotations
    with open(os.path.join(data_root_dir, f"../few_shot_{each_class_cnt}.json"), 'w') as f:
        json.dump(annotations_dict, f)


def filter_data(ann_file, data_root_dir):
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
    confusion_mat = dict([(c, dict((c1, 0) for c1 in coco.cats.keys())) for c in coco.cats.keys()])
    for img_id in tqdm.tqdm(img_ids[:]):

        ann_ids = coco.getAnnIds([img_id])
        anns = coco.loadAnns(ann_ids)
        if len(anns) == 0:
            continue

        for ann in anns:

            mask = coco.annToMask(ann)
            new_polyes = mask2poly([mask])
            aa = np.min(new_polyes, axis=0).astype(int).tolist()
            x1, y1 = np.min(new_polyes[0], axis=0).astype(int).tolist()
            x2, y2 = np.max(new_polyes[0], axis=0).astype(int).tolist()
            bbox = [x1, y1, x2 - x1, y2 - y1]
            ann['bbox'] = bbox
            new_gt_polys = [mask.flatten().astype(int).tolist() for mask in new_polyes]
            ann['segmentation'] = new_gt_polys

            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 80 or max(w, h) < 12:
                continue
            # new_annotations.append()
            new_annotations.append(ann)

    annotations_dict['annotations'] = new_annotations
    with open(os.path.join(data_root_dir, f"../filtered_{500}.json"), 'w') as f:
        json.dump(annotations_dict, f)

    # annotations_dict['categories'] = coco_find_grin.dataset['categories']
    annotations_dict['images'] = coco.loadImgs(new_images)
    # annotations_dict['annotations'] = new_annotations


def main():
    data_root_dir = "/mnt/data/datasets/gaofen/FAIR1M2.0/fair1_1000/val1000/images"
    annotation_file = "/mnt/data/datasets/gaofen/FAIR1M2.0/fair1_1000/val1000/few_shot_500.json"
    filter_data(annotation_file, data_root_dir)
    # data_root_dir = "/mnt/data/datasets/gaofen/FAIR1M2.0/fair1_1000/train1000/images"
    # annotation_file = "/mnt/data/datasets/gaofen/FAIR1M2.0/fair1_1000/train1000/train1000_5classes.json"

    # create_few_shot(annotation_file,
    #                 data_root_dir, 500)
    load = True
    if load:
        data_root_dir = "/mnt/data/datasets/gaofen/FAIR1M2.0/fair1_1000/train1000/images"
        annotation_file = "/mnt/data/datasets/gaofen/FAIR1M2.0/fair1_1000/train1000/few_shot_8.json"
        load_annotations(annotation_file, data_root_dir)


if __name__ == '__main__':
    main()
