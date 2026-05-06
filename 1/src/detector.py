import numpy as np
import onnxruntime as ort
import cv2


class YoloFireSmokeDetector:
    """
    YOLOv8n fire/smoke detector backed by an ONNX Runtime CPU session.

    Expected ONNX export (ultralytics, simplify=True):
        Input  shape: [1, 3, imgsz, imgsz]
        Output shape: [1, 4+nc, num_boxes]
          rows 0-3 : cx, cy, w, h  (pixel-space, relative to imgsz)
          rows 4.. : class scores  (sigmoid already applied)
    """

    CLASS_NAMES = ['fire', 'smoke']

    def __init__(
        self,
        model_path: str,
        imgsz: int = 320,
        conf_thres: float = 0.35,
        iou_thres: float = 0.45,
    ):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4   # use all 4 Pi cores
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session   = ort.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])
        self.imgsz     = imgsz
        self.conf_thres = conf_thres
        self.iou_thres  = iou_thres
        self._input_name = self.session.get_inputs()[0].name

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _letterbox(self, img: np.ndarray, size: int):
        """Resize keeping aspect ratio, pad to square.  Returns (img, scale, (dw, dh))."""
        h, w = img.shape[:2]
        scale = size / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

        dw = (size - nw) / 2
        dh = (size - nh) / 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img, scale, (left, top)

    def _preprocess(self, frame: np.ndarray):
        """BGR frame → NCHW float32 tensor + letterbox metadata."""
        lb, scale, pad = self._letterbox(frame, self.imgsz)
        rgb = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = np.expand_dims(rgb.transpose(2, 0, 1), 0)   # NCHW
        return tensor, scale, pad

    def _postprocess(self, raw: np.ndarray, orig_hw: tuple, scale: float, pad: tuple):
        """
        raw   : shape [1, 4+nc, num_boxes]
        returns: list of dicts {bbox, class_id, class_name, confidence}
        """
        raw = raw[0].T                      # [num_boxes, 4+nc]
        nc  = len(self.CLASS_NAMES)
        boxes_cxcywh = raw[:, :4]
        scores       = raw[:, 4:4 + nc]    # [num_boxes, nc]

        class_ids   = scores.argmax(axis=1)
        confidences = scores.max(axis=1)

        mask = confidences >= self.conf_thres
        boxes_cxcywh = boxes_cxcywh[mask]
        confidences  = confidences[mask]
        class_ids    = class_ids[mask]

        if len(boxes_cxcywh) == 0:
            return []

        # cx,cy,w,h (imgsz space) → x1,y1,x2,y2 (imgsz space)
        cx, cy, w, h = boxes_cxcywh.T
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        xyxy_imgsz = np.stack([x1, y1, x2, y2], axis=1)

        # Remove letterbox padding and rescale to original frame
        pad_x, pad_y = pad
        xyxy_imgsz[:, [0, 2]] -= pad_x
        xyxy_imgsz[:, [1, 3]] -= pad_y
        xyxy_orig = xyxy_imgsz / scale

        orig_h, orig_w = orig_hw
        xyxy_orig[:, [0, 2]] = xyxy_orig[:, [0, 2]].clip(0, orig_w)
        xyxy_orig[:, [1, 3]] = xyxy_orig[:, [1, 3]].clip(0, orig_h)

        # NMS per-class
        results = []
        for cls in range(nc):
            idx = class_ids == cls
            if not idx.any():
                continue
            b = xyxy_orig[idx].tolist()
            c = confidences[idx].tolist()
            keep = cv2.dnn.NMSBoxes(b, c, self.conf_thres, self.iou_thres)
            if len(keep) == 0:
                continue
            for k in keep:
                i = int(k)
                results.append({
                    'bbox':       [round(v, 1) for v in b[i]],
                    'class_id':   cls,
                    'class_name': self.CLASS_NAMES[cls],
                    'confidence': round(float(c[i]), 4),
                })

        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list:
        """Detect fire/smoke in a BGR frame.  Returns list of detection dicts."""
        tensor, scale, pad = self._preprocess(frame)
        outputs = self.session.run(None, {self._input_name: tensor})
        return self._postprocess(outputs[0], frame.shape[:2], scale, pad)
