import cv2
import numpy as np
import onnxruntime as ort

ort.set_default_logger_severity(3)


def _distance2bbox(points, distance):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points, distance):
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


class SCRFD:
    input_mean = 127.5
    input_std = 128.0

    def __init__(self, model_file, provider="CPUExecutionProvider"):
        self.session = ort.InferenceSession(model_file, providers=[provider])
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.center_cache = {}

        self.nms_thresh = 0.4
        self.det_thresh = 0.5
        self.input_size = (640, 640)

        n = len(self.output_names)
        self.use_kps = False
        self._num_anchors = 1
        if n == 6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif n == 9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif n == 10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif n == 15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True
        else:
            raise ValueError(f"unexpected SCRFD output count: {n}")

    def prepare(self, input_size=(640, 640), det_thresh=0.5, nms_thresh=0.4):
        self.input_size = tuple(input_size)
        self.det_thresh = det_thresh
        self.nms_thresh = nms_thresh
        self.center_cache.clear()

    def _forward(self, det_img):
        input_w, input_h = self.input_size
        blob = cv2.dnn.blobFromImage(
            det_img,
            1.0 / self.input_std,
            (input_w, input_h),
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=True,
        )
        net_outs = self.session.run(self.output_names, {self.input_name: blob})

        scores_list = []
        bboxes_list = []
        kpss_list = []
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = net_outs[idx]
            bbox_preds = net_outs[idx + self.fmc] * stride
            kps_preds = net_outs[idx + self.fmc * 2] * stride if self.use_kps else None

            height = input_h // stride
            width = input_w // stride
            key = (height, width, stride)
            anchor_centers = self.center_cache.get(key)
            if anchor_centers is None:
                anchor_centers = np.stack(
                    np.mgrid[:height, :width][::-1], axis=-1
                ).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack(
                        [anchor_centers] * self._num_anchors, axis=1
                    ).reshape((-1, 2))
                self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= self.det_thresh)[0]
            bboxes = _distance2bbox(anchor_centers, bbox_preds)
            scores_list.append(scores[pos_inds])
            bboxes_list.append(bboxes[pos_inds])
            if self.use_kps:
                kpss = _distance2kps(anchor_centers, kps_preds).reshape((-1, 5, 2))
                kpss_list.append(kpss[pos_inds])

        if not scores_list or sum(s.size for s in scores_list) == 0:
            return None, None

        scores = np.vstack(scores_list).ravel()
        order = scores.argsort()[::-1]
        bboxes = np.vstack(bboxes_list)
        kpss = np.vstack(kpss_list) if self.use_kps else None

        pre_det = np.hstack((bboxes, scores[:, None])).astype(np.float32)
        pre_det = pre_det[order, :]
        if kpss is not None:
            kpss = kpss[order, :, :]
        return pre_det, kpss

    def detect(self, img, max_num=0):
        input_w, input_h = self.input_size
        im_ratio = img.shape[0] / img.shape[1]
        model_ratio = input_h / input_w
        if im_ratio > model_ratio:
            new_h = input_h
            new_w = int(new_h / im_ratio)
        else:
            new_w = input_w
            new_h = int(new_w * im_ratio)
        det_scale = new_h / img.shape[0]

        resized = cv2.resize(img, (new_w, new_h))
        det_img = np.zeros((input_h, input_w, 3), dtype=np.uint8)
        det_img[:new_h, :new_w, :] = resized

        pre_det, kpss = self._forward(det_img)
        if pre_det is None:
            kps_empty = np.empty((0, 5, 2), dtype=np.float32) if self.use_kps else None
            return np.empty((0, 5), dtype=np.float32), kps

        pre_det[:, :4] /= det_scale
        if kpss is not None:
            kpss = kpss / det_scale

        keep = self._nms(pre_det)
        det = pre_det[keep, :]
        if kpss is not None:
            kpss = kpss[keep, :, :]
        if max_num > 0 and det.shape[0] > max_num:
            det, kpss = self._keep_centered(det, kpss, img, max_num)
        return det, kpss

    def _nms(self, dets):
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]
        return keep

    @staticmethod
    def _keep_center(det, kpss, img, max_num):
        area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
        cx, cy = img.shape[1] // 2, img.shape[0] // 2
        offs = np.vstack(
            [(det[:, 0] + det[:, 2]) / 2 - cx, (det[:, 1] + det[:, 3]) / 2 - cy]
        )
        dist = np.sum(np.power(offs, 2.0), 0)
        values = area - dist * 2.0
        index = values.argsort()[::-1][:max_num]
        det = det[index, :]
        if kpss is not None:
            kpss = kpss[index, :, :]
        return det, kpss
