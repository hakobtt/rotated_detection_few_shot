import torch
import numpy as np
import cv2
from mmcv.runner import load_checkpoint
from mmcv.runner.fp16_utils import force_fp32
from mmdet.core import bbox2roi, multi_apply
from mmdet.models import DETECTORS, build_detector
from mmdet_custom.apis import draw_poly_detections
from mmdet_custom.core import mask2poly, get_best_begin_point

from ssod.utils.structure_utils import dict_split, weighted_loss
from ssod.utils import log_image_with_boxes, log_every_n

from .multi_stream_detector import MultiSteamDetector
from .utils import Transform2D, filter_invalid
import pycocotools.mask as maskUtils
from mmdet.core.mask.structures import BitmapMasks


def masks_to_bbox(masks):
    if masks.shape[0] > 0:
        res = torch.stack(
            [
                torch.min(masks[:, :-1:2], dim=1)[0],
                torch.min(masks[:, 1:-1:2], dim=1)[0],
                torch.max(masks[:, :-1:2], dim=1)[0],
                torch.max(masks[:, 1:-1:2], dim=1)[0],
                masks[:, -1],
            ], dim=-1)

        return res
    else:
        return masks.new_zeros(0, 5)


def poly2mask(mask_ann, img_h, img_w):
    rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
    rle = maskUtils.merge(rles)
    mask = maskUtils.decode(rle)
    return mask


def make_masks(img_polys, img_meta):
    img_h, img_w = img_meta["pad_shape"][:2]
    img_masks = BitmapMasks(
        [poly2mask([mask], img_h, img_w) for mask in img_polys.detach().cpu().numpy()],
        img_h, img_w
    )
    return img_masks


@DETECTORS.register_module()
class SoftTeacher(MultiSteamDetector):
    def __init__(self, model: dict, train_cfg=None, test_cfg=None):
        self.CLASSES = ["airplane", "ship", "vehicle", "court", "road"]
        teacher = build_detector(
            model,
        )
        student = build_detector(
            model,
        )

        super(SoftTeacher, self).__init__(
            dict(
                teacher=teacher,
                student=student
            ),
            train_cfg=train_cfg,
            test_cfg=test_cfg,
        )
        if train_cfg is not None:
            self.freeze("teacher")
            self.unsup_weight = self.train_cfg.unsup_weight

    def forward_train(self, img, img_metas, **kwargs):
        super().forward_train(img, img_metas, **kwargs)
        kwargs.update({"img": img})
        kwargs.update({"img_metas": img_metas})
        kwargs.update({"tag": [meta["tag"] for meta in img_metas]})
        data_groups = dict_split(kwargs, "tag")
        for _, v in data_groups.items():
            v.pop("tag")

        loss = {}
        # ! Warnings: By splitting losses for supervised data and unsupervised data with different names,
        # ! it means that at least one sample for each group should be provided on each gpu.
        # ! In some situation, we can only put one image per gpu, we have to return the sum of loss
        # ! and log the loss with logger instead. Or it will try to sync tensors don't exist.
        if "sup" in data_groups:
            gt_bboxes = data_groups["sup"]["gt_bboxes"]
            log_every_n(
                {"sup_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
            )
            sup_loss = self.student.forward_train(**data_groups["sup"])
            sup_loss = {"sup_" + k: v for k, v in sup_loss.items()}
            loss.update(**sup_loss)
        if "unsup_student" in data_groups:
            unsup_loss = weighted_loss(
                self.foward_unsup_train(
                    data_groups["unsup_teacher"], data_groups["unsup_student"]
                ),
                weight=self.unsup_weight,
            )
            unsup_loss = {"unsup_" + k: v for k, v in unsup_loss.items()}
            loss.update(**unsup_loss)

        return loss

    def foward_unsup_train(self, teacher_data, student_data):
        # sort the teacher and student input to avoid some bugs
        tnames = [meta["filename"] for meta in teacher_data["img_metas"]]
        snames = [meta["filename"] for meta in student_data["img_metas"]]
        tidx = [tnames.index(name) for name in snames]
        with torch.no_grad():
            teacher_info = self.extract_teacher_info(
                teacher_data["img"][
                    torch.Tensor(tidx).to(teacher_data["img"].device).long()
                ],
                [teacher_data["img_metas"][idx] for idx in tidx],
                [teacher_data["proposals"][idx] for idx in tidx]
                if ("proposals" in teacher_data)
                   and (teacher_data["proposals"] is not None)
                else None,
            )
        student_info = self.extract_student_info(**student_data)

        return self.compute_pseudo_label_loss(student_info, teacher_info, teacher_data, student_data)

    def compute_pseudo_label_loss(self, student_info, teacher_info, teacher_data, student_data):
        M = self._get_trans_mat(
            teacher_info["transform_matrix"], student_info["transform_matrix"]
        )

        pseudo_bboxes = self._transform_bbox(
            teacher_info["det_bboxes"],
            M,
            [meta["img_shape"] for meta in student_info["img_metas"]],
        )
        pseudo_masks = self._transform_polys(
            teacher_info["det_masks"],
            M,
            [meta["img_shape"][:2] for meta in student_info["img_metas"]],
        )
        pseudo_labels = teacher_info["det_labels"]
        assert len(pseudo_masks) == len(pseudo_bboxes) == len(pseudo_labels)
        img_cnt = len(pseudo_labels)
        pseudo_masks = [make_masks(polys, img_meta)
                        for polys, img_meta in zip(pseudo_masks, student_info["img_metas"])
                        ]
        for i in range(img_cnt):
            mask_arrays = pseudo_masks[i].masks
            if mask_arrays.shape[0] == 0:
                continue

            valid_masks = np.any(mask_arrays.reshape((mask_arrays.shape[0], -1)), axis=1)
            pseudo_bboxes[i] = pseudo_bboxes[i][valid_masks]
            pseudo_masks[i] = pseudo_masks[i][valid_masks]
            pseudo_labels[i] = pseudo_labels[i][valid_masks]

        img = student_data["img"]

        # for i in range(len(img)):
        #
        #     std = torch.tensor([58.395, 57.12, 57.375]).to(img.device)
        #     mean = torch.tensor([123.675, 116.28, 103.53]).to(img.device)
        #     img_0 = img[i].permute((1, 2, 0))
        #     img_0 = img_0 * std + mean
        #     img_0 = img_0.detach().cpu().numpy().astype(np.uint8).copy()
        #     img_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2RGB)
        #     # masks = = proposal_list[0].detach().cpu().numpy()
        #     masks = pseudo_masks[i]
        #     boxes = pseudo_bboxes[i].detach().cpu().numpy()
        #
        #     pos_gt_polys = mask2poly(masks)
        #     pos_gt_bp_polys = get_best_begin_point(pos_gt_polys)
        #     color = (255, 0, 0)
        #     for bbox in boxes:
        #         bbox = bbox.astype(int).flatten()
        #         cv2.rectangle(img_0, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        #     for bbox in pos_gt_bp_polys:
        #         bbox = bbox.astype(int).flatten()
        #         for i in range(3):
        #             cv2.line(img_0, (bbox[i * 2], bbox[i * 2 + 1]), (bbox[(i + 1) * 2], bbox[(i + 1) * 2 + 1]),
        #                      color=color,
        #                      thickness=2, lineType=cv2.LINE_AA)
        #         cv2.line(img_0, (bbox[6], bbox[7]), (bbox[0], bbox[1]), color=color, thickness=2, lineType=cv2.LINE_AA)
        #
        #     cv2.imshow("res_img", img_0)
        #     cv2.waitKey(0)

        loss = {}
        rpn_loss, proposal_list = self.rpn_loss(
            student_info["rpn_out"],
            pseudo_bboxes,
            student_info["img_metas"],
            student_info=student_info,
        )
        loss.update(rpn_loss)
        if proposal_list is not None:
            student_info["proposals"] = proposal_list
        if self.train_cfg.use_teacher_proposal:
            proposals = self._transform_bbox(
                teacher_info["proposals"],
                M,
                [meta["img_shape"] for meta in student_info["img_metas"]],
            )
        else:
            proposals = student_info["proposals"]

        loss.update(
            self.unsup_rcnn_cls_loss(
                img,
                student_info["backbone_feature"],
                student_info["img_metas"],
                proposals,
                pseudo_bboxes,
                pseudo_masks,
                pseudo_labels,
                teacher_info["transform_matrix"],
                student_info["transform_matrix"],
                teacher_info["img_metas"],
                teacher_info["backbone_feature"],
                student_info=student_info,
            )
        )
        loss.update(
            self.unsup_rcnn_reg_loss(
                img,
                student_info["backbone_feature"],
                student_info["img_metas"],
                proposals,
                pseudo_bboxes,
                pseudo_masks,
                pseudo_labels,
                student_info=student_info,
            )
        )
        return loss

    def rpn_loss(
            self,
            rpn_out,
            pseudo_bboxes,
            img_metas,
            gt_bboxes_ignore=None,
            student_info=None,
            **kwargs,
    ):
        if self.student.with_rpn:
            gt_bboxes = []
            for bbox in pseudo_bboxes:
                bbox, _, _ = filter_invalid(
                    bbox[:, :4],
                    score=bbox[
                          :, 4
                          ],  # TODO: replace with foreground score, here is classification score,
                    thr=self.train_cfg.rpn_pseudo_threshold,
                    min_size=self.train_cfg.min_pseduo_box_size,
                )
                gt_bboxes.append(bbox)
            log_every_n(
                {"rpn_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
            )
            loss_inputs = rpn_out + [[bbox.float() for bbox in gt_bboxes], img_metas]
            losses = self.student.rpn_head.loss(
                *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore
            )
            proposal_cfg = self.student.train_cfg.get(
                "rpn_proposal", self.student.test_cfg.rpn
            )
            proposal_list = self.student.rpn_head.get_bboxes(
                *rpn_out, img_metas=img_metas, cfg=proposal_cfg
            )
            # log_image_with_boxes(
            #     "rpn",
            #     student_info["img"][0],
            #     pseudo_bboxes[0][:, :4],
            #     bbox_tag="rpn_pseudo_label",
            #     scores=pseudo_bboxes[0][:, 4],
            #     interval=500,
            #     img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
            # )
            return losses, proposal_list
        else:
            return {}, None

    def unsup_rcnn_cls_loss(
            self,
            img,
            feat,
            img_metas,
            proposal_list,
            pseudo_bboxes,
            pseudo_masks,
            pseudo_labels,
            teacher_transMat,
            student_transMat,
            teacher_img_metas,
            teacher_feat,
            student_info=None,
            **kwargs,
    ):
        gt_bboxes, gt_labels, pseudo_masks = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
            pseudo_labels,
            [bbox[:, 4] for bbox in pseudo_bboxes],
            pseudo_masks,
            thr=self.train_cfg.cls_pseudo_threshold,
        )
        log_every_n(
            {"rcnn_cls_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
        )
        sampling_results = self.get_sampling_result(
            img_metas,
            proposal_list,
            gt_bboxes,
            gt_labels,
        )
        selected_bboxes = [res.bboxes[:, :4] for res in sampling_results]
        rois = bbox2roi(selected_bboxes)
        bbox_results = self.student.roi_head._bbox_forward(feat, rois)
        bbox_targets = self.student.roi_head.bbox_head.get_targets(
            [None] * len(img),
            sampling_results, pseudo_masks, gt_labels, self.student.train_cfg.rcnn
        )
        M = self._get_trans_mat(student_transMat, teacher_transMat)
        aligned_proposals = self._transform_bbox(
            selected_bboxes,
            M,
            [meta["img_shape"] for meta in teacher_img_metas],
        )
        with torch.no_grad():
            _, _scores = self.teacher.roi_head.simple_test_bboxes(
                teacher_feat,
                teacher_img_metas,
                aligned_proposals,
                None,
                rescale=False,
            )
            bg_score = torch.cat([_score[:, -1] for _score in _scores])
            assigned_label, _, _, _ = bbox_targets
            neg_inds = assigned_label == self.student.roi_head.bbox_head.num_classes
            bbox_targets[1][neg_inds] = bg_score[neg_inds].detach()
        loss = self.student.roi_head.bbox_head.loss(
            bbox_results["cls_score"],
            bbox_results["bbox_pred"],
            rois,
            *bbox_targets,
            reduction_override="none",
        )
        loss["rbbox_loss_cls"] = loss["rbbox_loss_cls"].sum() / max(bbox_targets[1].sum(), 1.0)
        loss["rbbox_loss_bbox"] = loss["rbbox_loss_bbox"].sum() / max(
            bbox_targets[1].size()[0], 1.0
        )
        # if len(gt_bboxes[0]) > 0:
        #     log_image_with_boxes(
        #         "rcnn_cls",
        #         student_info["img"][0],
        #         gt_bboxes[0],
        #         bbox_tag="pseudo_label",
        #         labels=gt_labels[0],
        #         class_names=self.CLASSES,
        #         interval=500,
        #         img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
        #     )
        return loss

    def unsup_rcnn_reg_loss(
            self,
            img,
            feat,
            img_metas,
            proposal_list,
            pseudo_bboxes,
            pseudo_masks,
            pseudo_labels,
            student_info=None,
            **kwargs,
    ):
        gt_bboxes, gt_labels, gt_masks = multi_apply(
            filter_invalid,
            [bbox[:, :4] for bbox in pseudo_bboxes],
            pseudo_labels,
            [-bbox[:, 5:].mean(dim=-1) for bbox in pseudo_bboxes],
            pseudo_masks,
            thr=-self.train_cfg.reg_pseudo_threshold,
        )
        log_every_n(
            {"rcnn_reg_gt_num": sum([len(bbox) for bbox in gt_bboxes]) / len(gt_bboxes)}
        )
        rbbox_loss_bbox = self.student.roi_head.forward_train(img,
                                                              feat, img_metas, proposal_list, gt_bboxes, gt_labels,
                                                              gt_masks=gt_masks, **kwargs
                                                              )["rbbox_loss_bbox"]
        # if len(gt_bboxes[0]) > 0:
        #     log_image_with_boxes(
        #         "rcnn_reg",
        #         student_info["img"][0],
        #         gt_bboxes[0],
        #         bbox_tag="pseudo_label",
        #         labels=gt_labels[0],
        #         class_names=self.CLASSES,
        #         interval=500,
        #         img_norm_cfg=student_info["img_metas"][0]["img_norm_cfg"],
        #     )
        return {"rbbox_loss_bbox": rbbox_loss_bbox}

    def get_sampling_result(
            self,
            img_metas,
            proposal_list,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore=None,
            **kwargs,
    ):
        num_imgs = len(img_metas)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        sampling_results = []
        for i in range(num_imgs):
            assign_result = self.student.roi_head.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i], gt_labels[i]
            )
            sampling_result = self.student.roi_head.bbox_sampler.sample(
                assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
            )
            sampling_results.append(sampling_result)
        return sampling_results

    @force_fp32(apply_to=["bboxes", "trans_mat"])
    def _transform_bbox(self, bboxes, trans_mat, max_shape):
        bboxes = Transform2D.transform_bboxes(bboxes, trans_mat, max_shape)
        return bboxes

    @force_fp32(apply_to=["polys", "trans_mat"])
    def _transform_polys(self, poly, trans_mat, max_shape):
        poly = Transform2D.transform_polygon(poly, trans_mat, max_shape)
        return poly

    @force_fp32(apply_to=["mask", "trans_mat"])
    def _transform_masks(self, masks, trans_mat, max_shape):
        bboxes = Transform2D.transform_masks(masks, trans_mat, max_shape)
        return bboxes

    @force_fp32(apply_to=["a", "b"])
    def _get_trans_mat(self, a, b):
        return [bt @ at.inverse() for bt, at in zip(b, a)]

    def extract_student_info(self, img, img_metas, proposals=None, **kwargs):
        student_info = {}
        student_info["img"] = img
        feat = self.student.extract_feat(img)
        student_info["backbone_feature"] = feat
        if self.student.with_rpn:
            rpn_out = self.student.rpn_head(feat)
            student_info["rpn_out"] = list(rpn_out)
        student_info["img_metas"] = img_metas
        student_info["proposals"] = proposals
        student_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
            for meta in img_metas
        ]
        return student_info

    def show_inf(self, img, img_metas, ):
        # load_checkpoint(self.teacher, self.checkpoint_path, )

        # polygon_res = []
        # for rr in result[0]:
        #     poly = RotBox2Polys(rr)
        #     polygon_res.append(np.concatenate([poly, rr[:, -1:]], axis=1))
        # result = polygon_res

        for i in range(len(img)):
            data = dict(img=[img[i:i + 1]], img_metas=[img_metas[i:i + 1]])
            with torch.no_grad():
                result = self.student(return_loss=False, rescale=True, **data)
            result = result[0]
            std = torch.tensor([58.395, 57.12, 57.375]).to(img.device)
            mean = torch.tensor([123.675, 116.28, 103.53]).to(img.device)
            img_0 = data["img"][0][0].permute((1, 2, 0))
            img_0 = img_0 * std + mean
            img_0 = img_0.detach().cpu().numpy().astype(np.uint8).copy()
            img_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2RGB)

            img_0 = draw_poly_detections(img_0, result, ["", "airplane", "ship", "vehicle", "court", "road"], scale=1,
                                         threshold=0.2,
                                         )
            cv2.imshow("img_res_hakob", img_0)
            cv2.waitKey(0)

    def extract_teacher_info(self, img, img_metas, proposals=None, **kwargs):

        # self.show_inf(img, img_metas)
        teacher_info = {}
        feat = self.teacher.extract_feat(img)
        teacher_info["backbone_feature"] = feat
        if proposals is None:
            proposal_cfg = self.teacher.train_cfg.get(
                "rpn_proposal", self.teacher.test_cfg.rpn
            )
            rpn_out = list(self.teacher.rpn_head(feat))
            proposal_list = self.teacher.rpn_head.get_bboxes(
                *rpn_out, img_metas=img_metas, cfg=proposal_cfg
            )
        else:
            proposal_list = proposals

        teacher_info["proposals"] = proposal_list

        proposal_list, proposal_label_list = self.teacher.roi_head.simple_test_bboxes(
            feat, img_metas, proposal_list, self.teacher.test_cfg.rcnn, rescale=False
        )

        proposal_masks = [p.to(feat[0].device) for p in proposal_list]

        proposal_list = [
            masks_to_bbox(p) for p in proposal_masks
        ]
        proposal_masks = [img_polys[:, :-1]
                          for img_polys, img_meta in zip(proposal_masks, img_metas)
                          ]

        proposal_list = [
            p if p.shape[0] > 0 else p.new_zeros(0, 5) for p in proposal_list
        ]
        proposal_label_list = [p.to(feat[0].device) for p in proposal_label_list]
        # filter invalid box roughly
        if isinstance(self.train_cfg.pseudo_label_initial_score_thr, float):
            thr = self.train_cfg.pseudo_label_initial_score_thr
        else:
            # TODO: use dynamic threshold
            raise NotImplementedError("Dynamic Threshold is not implemented yet.")
        proposal_list, proposal_label_list, proposal_masks = list(
            zip(
                *[
                    filter_invalid(
                        proposal,
                        proposal_label,
                        proposal[:, -1],
                        prop_mask,
                        thr=thr,
                        min_size=self.train_cfg.min_pseduo_box_size,
                    )
                    for proposal, proposal_label, prop_mask in zip(
                        proposal_list, proposal_label_list, proposal_masks
                    )
                ]
            )
        )

        det_bboxes = proposal_list
        det_masks = proposal_masks
        reg_unc = self.compute_uncertainty_with_aug(
            feat, img_metas, proposal_list, proposal_label_list
        )
        det_bboxes = [
            torch.cat([bbox, unc], dim=-1) for bbox, unc in zip(det_bboxes, reg_unc)
        ]
        det_labels = list(proposal_label_list)
        teacher_info["det_bboxes"] = det_bboxes
        teacher_info["det_masks"] = det_masks
        teacher_info["det_labels"] = det_labels
        teacher_info["transform_matrix"] = [
            torch.from_numpy(meta["transform_matrix"]).float().to(feat[0][0].device)
            for meta in img_metas
        ]
        teacher_info["img_metas"] = img_metas
        return teacher_info

    def compute_uncertainty_with_aug(
            self, feat, img_metas, proposal_list, proposal_label_list
    ):
        auged_proposal_list = self.aug_box(
            proposal_list, self.train_cfg.jitter_times, self.train_cfg.jitter_scale
        )
        # flatten
        auged_proposal_list = [
            auged.reshape(-1, auged.shape[-1]) for auged in auged_proposal_list
        ]

        bboxes, _ = self.teacher.roi_head.simple_test_bboxes(
            feat,
            img_metas,
            auged_proposal_list,
            None,
            rescale=False,
        )
        bboxes = [
            masks_to_bbox(p)[:, :4] for p in bboxes
        ]
        reg_channel = max([bbox.shape[-1] for bbox in bboxes]) // 4
        bboxes = [
            bbox.reshape(self.train_cfg.jitter_times, -1, bbox.shape[-1])
            if bbox.numel() > 0
            else bbox.new_zeros(self.train_cfg.jitter_times, 0, 4 * reg_channel).float()
            for bbox in bboxes
        ]

        box_unc = [bbox.std(dim=0) for bbox in bboxes]
        bboxes = [bbox.mean(dim=0) for bbox in bboxes]
        # scores = [score.mean(dim=0) for score in scores]
        if reg_channel != 1:
            bboxes = [
                bbox.reshape(bbox.shape[0], reg_channel, 4)[
                    torch.arange(bbox.shape[0]), label
                ]
                for bbox, label in zip(bboxes, proposal_label_list)
            ]
            box_unc = [
                unc.reshape(unc.shape[0], reg_channel, 4)[
                    torch.arange(unc.shape[0]), label
                ]
                for unc, label in zip(box_unc, proposal_label_list)
            ]

        box_shape = [(bbox[:, 2:4] - bbox[:, :2]).clamp(min=1.0) for bbox in bboxes]
        # relative unc
        box_unc = [
            unc / wh[:, None, :].expand(-1, 2, 2).reshape(-1, 4)
            if wh.numel() > 0
            else unc
            for unc, wh in zip(box_unc, box_shape)
        ]
        return box_unc

    @staticmethod
    def aug_box(boxes, times=1, frac=0.06):
        def _aug_single(box):
            # random translate
            # TODO: random flip or something
            box_scale = box[:, 2:4] - box[:, :2]
            box_scale = (
                box_scale.clamp(min=1)[:, None, :].expand(-1, 2, 2).reshape(-1, 4)
            )
            aug_scale = box_scale * frac  # [n,4]

            offset = (
                    torch.randn(times, box.shape[0], 4, device=box.device)
                    * aug_scale[None, ...]
            )
            new_box = box.clone()[None, ...].expand(times, box.shape[0], -1)
            return torch.cat(
                [new_box[:, :, :4].clone() + offset, new_box[:, :, 4:]], dim=-1
            )

        return [_aug_single(box) for box in boxes]

    def _load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
    ):
        if not any(["student" in key or "teacher" in key for key in state_dict.keys()]):
            keys = list(state_dict.keys())
            state_dict.update({"teacher." + k: state_dict[k] for k in keys})
            state_dict.update({"student." + k: state_dict[k] for k in keys})
            for k in keys:
                state_dict.pop(k)

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
