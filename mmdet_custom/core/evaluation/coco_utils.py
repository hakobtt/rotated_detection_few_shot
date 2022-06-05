import mmcv
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .recall import eval_recalls


def coco_eval(result_file, result_types, coco, max_dets=(100, 300, 1000)):
    for res_type in result_types:
        assert res_type in [
            'proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'
        ]

    if mmcv.is_str(coco):
        coco = COCO(coco)
    assert isinstance(coco, COCO)

    if result_types == ['proposal_fast']:
        ar = fast_eval_recall(result_file, coco, np.array(max_dets))
        for i, num in enumerate(max_dets):
            print('AR@{}\t= {:.4f}'.format(num, ar[i]))
        return

    assert result_file.endswith('.json')
    coco_dets = coco.loadRes(result_file)

    img_ids = coco.getImgIds()
    import os
    import cv2
    import random
    data_root_dir = "/mnt/data/datasets/gaofen/FAIR1M2.0/fair1_1000/val_small_set1000/images"
    # cats = coco_dets.cats
    cats = coco.cats
    print(cats)
    show = False

    if show:
        for img_id in img_ids[::]:
            info = coco.loadImgs([img_id])[0]

            img_path = os.path.join(data_root_dir, info['file_name'])

            ann_ids = coco.getAnnIds([img_id])
            anns = coco.loadAnns(ann_ids)
            img = cv2.imread(img_path)
            for ann in anns:
                bbox = ann['bbox']
                x, y = int(bbox[0]), int(bbox[1])
                pts = np.array(ann['segmentation'], np.int32).reshape((4, 2))
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(img, [pts], True, [255, 255, 0], 2)
                text = str(cats[ann['category_id']]['name'])

                cv2.putText(img, text, (x, y),
                            cv2.QT_FONT_NORMAL, 0.5, (128, 128, 255))
            cv2.imshow('img', img)
            if cv2.waitKey(0) == 27:
                exit()

    for res_type in result_types:

        iou_type = 'bbox' if res_type == 'proposal' else res_type

        cocoEval = COCOeval(coco, coco_dets, iou_type)
        cocoEval.params.imgIds = img_ids

        # cocoEval._prepare()
        # # cocoEval.evaluate()
        # class_map = {
        #     "airplane": ([1], list(range(1, 13))),
        #     "ship": ([2], list(range(13, 22))),
        #     "vehicle": ([3], list(range(22, 32))),
        #     "court": ([4], list(range(32, 36))),
        #     "road": ([5], list(range(36, 39))),
        # }
        #
        # for class_name, (dt_cts, gt_cts) in class_map.items():
        #     sub_cat_statistics = {}  # sub_cat_id: [all_gt_count, all_det_count, all_tp_count]
        #     for img_id in img_ids:
        #         ious, dts, gts = cocoEval.computeIoUFineGrained(img_id, gt_cts, dt_cts)
        #
        #         for gt in gts:
        #             if gt['category_id'] not in sub_cat_statistics:
        #                 sub_cat_statistics[gt['category_id']] = [0, 0, 0]
        #             sub_cat_statistics[gt['category_id']][0] += 1
        #         if len(ious) == 0:
        #             continue
        #         # for dt in dts:
        #         #     if gt['category_id'] not in sub_cat_statistics:
        #         #         sub_cat_statistics = [0, 0, 0]
        #         #     sub_cat_statistics[gt['category_id']][1] += 1
        #         iou_thresh = 0.5
        #         score_thresh = 0.2
        #         ious = ious.transpose()
        #         for gt_idx, gt_ious in enumerate(ious):
        #             for det_idx, iou in enumerate(gt_ious):
        #                 if iou > iou_thresh and dts[det_idx]['score'] > score_thresh:
        #                     cat_id = gts[gt_idx]['category_id']
        #                     sub_cat_statistics[cat_id][2] += 1
        #                     break
        #     print(f"================={class_name}==================")
        #     print(f"{'class_name':<20}  "
        #           f"{'gt_count' :<10} | "
        #           f"{'tp_count':<10}  | "
        #           f"{'percent':<10}  | "
        #           )
        #     sub_cat_statistics = dict((k, sub_cat_statistics[k]) for k in sorted(sub_cat_statistics.keys()))
        #     for sub_cat_id, [gt_count, det_count, tp_count] in sub_cat_statistics.items():
        #         print(f"{cocoEval.cocoGt.cats[sub_cat_id]['name']:<20}  "
        #               f"{gt_count :<10} | "
        #               f"{tp_count:<10}  | "
        #               f"{f'{100 * tp_count / gt_count:.2f}%':<10}  | "
        #               )
        # exit()
        # cocoEval.accumulate()
        # cocoEval.summarize()

        for _, cat in cats.items():
            print(f"============================{cat['name']}================================")
            cocoEval.params.catIds = [cat['id'], ]
            if res_type == 'proposal':
                cocoEval.params.useCats = 0
                cocoEval.params.maxDets = list(max_dets)
            cocoEval.stats
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            print("akop")


def fast_eval_recall(results,
                     coco,
                     max_dets,
                     iou_thrs=np.arange(0.5, 0.96, 0.05)):
    if mmcv.is_str(results):
        assert results.endswith('.pkl')
        results = mmcv.load(results)
    elif not isinstance(results, list):
        raise TypeError(
            'results must be a list of numpy arrays or a filename, not {}'.
                format(type(results)))

    gt_bboxes = []
    img_ids = coco.getImgIds()
    for i in range(len(img_ids)):
        ann_ids = coco.getAnnIds(imgIds=img_ids[i])
        ann_info = coco.loadAnns(ann_ids)
        if len(ann_info) == 0:
            gt_bboxes.append(np.zeros((0, 4)))
            continue
        bboxes = []
        for ann in ann_info:
            if ann.get('ignore', False) or ann['iscrowd']:
                continue
            x1, y1, w, h = ann['bbox']
            bboxes.append([x1, y1, x1 + w - 1, y1 + h - 1])
        bboxes = np.array(bboxes, dtype=np.float32)
        if bboxes.shape[0] == 0:
            bboxes = np.zeros((0, 4))
        gt_bboxes.append(bboxes)

    recalls = eval_recalls(
        gt_bboxes, results, max_dets, iou_thrs, print_summary=False)
    ar = recalls.mean(axis=1)
    return ar


def xyxy2xywh(bbox):
    _bbox = bbox.tolist()
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] - _bbox[0] + 1,
        _bbox[3] - _bbox[1] + 1,
    ]


def segm_to_xywh(segm):
    min_x = np.min(segm[0::2])
    min_y = np.min(segm[1::2])
    max_x = np.max(segm[0::2])
    max_y = np.max(segm[1::2])
    return [
        min_x,
        min_y,
        max_x - min_x,
        max_y - min_y,
    ]


def proposal2json(dataset, results):
    json_results = []
    for idx in range(len(dataset)):
        img_id = dataset.img_ids[idx]
        bboxes = results[idx]
        for i in range(bboxes.shape[0]):
            data = dict()
            data['image_id'] = img_id
            data['bbox'] = xyxy2xywh(bboxes[i])
            data['score'] = float(bboxes[i][4])
            data['category_id'] = 1
            json_results.append(data)
    return json_results


def det2json(dataset, results):
    json_results = []
    for idx in range(len(dataset)):
        img_id = dataset.img_ids[idx]
        result = results[idx]
        for label in range(len(result)):
            bboxes = result[label]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = dataset.cat_ids[label]
                json_results.append(data)
    return json_results


def segm2json(dataset, results):
    json_results = []

    for idx in range(len(dataset)):
        img_id = dataset.img_ids[idx]
        dets = results[idx]
        for label in range(len(dets)):
            det_data = dets[label]

            for i in range(det_data.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = segm_to_xywh(det_data[i][:-1])
                data['score'] = float(det_data[i][-1])
                data['category_id'] = dataset.cat_ids[label - 1]
                # segms[i]['counts'] = segms[i]['counts'].decode()
                data['segmentation'] = [det_data[i].tolist()[:-1]]
                json_results.append(data)
    return json_results


def results2json(dataset, results, out_file):
    json_results = segm2json(dataset, results)
    # if isinstance(results[0], list):
    #     json_results = det2json(dataset, results)
    # elif isinstance(results[0], tuple):
    #     json_results = segm2json(dataset, results)
    # elif isinstance(results[0], np.ndarray):
    #     json_results = proposal2json(dataset, results)
    # else:
    #     raise TypeError('invalid type of results')
    mmcv.dump(json_results, out_file)
