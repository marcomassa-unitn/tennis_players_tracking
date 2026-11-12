"""Ball detector backends for the motion model: crops in, boxes out.

Every backend takes a batch of CROP x CROP crops built by
tracking.ball_online.pack_channels and returns, per crop, a list of
(x1, y1, x2, y2, conf) in crop pixels.

  UltralyticsBackend  .pt or TensorRT .engine weights, through ultralytics
  OnnxBackend         .onnx weights, through onnxruntime (CPU or CUDA), with its own
                      pre- and post-processing (no ultralytics or torch needed)
  DiffBlobBackend     no model at all: the ball-sized blobs of the difference channel.
                      The fallback without weights, and the stand-in used by the tests.

The crops are stored like OpenCV images, channels in BGR order. Ultralytics flips
them to RGB both when it trains on the PNGs and when it predicts on arrays, so the
network sees (gray, diff t, diff t-1); OnnxBackend makes the same flip.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

CROP = 320                 # the model's input size: crops at native resolution
DETECT_CONF = 0.15         # boxes below this are dropped
NMS_IOU = 0.3
GPU_BATCHES = (1, 2, 4, 8, 16, 32)   # batch sizes a GPU session sees (see OnnxBackend)
ENGINE_MAX_BATCH = 16      # crops per call to a TensorRT engine (exported for up to 32)


def batch_bucket(n, sizes=GPU_BATCHES):
    """Smallest of `sizes` holding n crops (the largest if none does)."""
    for size in sizes:
        if n <= size:
            return size
    return sizes[-1]


def load_backend(spec, conf=DETECT_CONF, device=None):
    """Backend for `spec`: a weights path (.pt/.engine/.onnx) or "diff" (no model)."""
    if spec in (None, "", "diff"):
        return DiffBlobBackend()
    ext = os.path.splitext(spec)[1].lower()
    if not os.path.exists(spec):
        raise FileNotFoundError(f"Ball model not found: {spec}")
    if ext == ".onnx":
        return OnnxBackend(spec, conf=conf, device=device)
    if ext == ".pt":
        return TorchBackend(spec, conf=conf, device=device)
    if ext == ".engine":
        return UltralyticsBackend(spec, conf=conf, device=device)
    raise ValueError(f"Unknown ball model format: {spec} (expected .pt, .engine or .onnx)")


def _nms(boxes, scores, conf, iou=NMS_IOU):
    if not len(boxes):
        return []
    xywh = [[float(x1), float(y1), float(x2 - x1), float(y2 - y1)] for x1, y1, x2, y2 in boxes]
    keep = cv2.dnn.NMSBoxes(xywh, [float(s) for s in scores], conf, iou)
    return [(*map(float, boxes[i]), float(scores[i])) for i in np.array(keep).ravel()]


class UltralyticsBackend:
    """YOLO weights through ultralytics (.pt on CPU/GPU, .engine on TensorRT)."""

    name = "ultralytics"

    def __init__(self, path, conf=DETECT_CONF, device=None):
        from ultralytics import YOLO
        self.model = YOLO(path, task="detect")
        self.conf = conf
        self.device = device
        self.half = path.endswith(".engine")
        self.detect([np.zeros((CROP, CROP, 3), np.uint8)])      # first call: model setup

    def detect(self, crops):
        if not len(crops):
            return []
        kw = {"device": self.device} if self.device is not None else {}
        out = []
        # A TensorRT engine only takes batches within its optimisation profile.
        step = ENGINE_MAX_BATCH if self.half else len(crops)
        crops = list(crops)
        for i in range(0, len(crops), step):
            results = self.model.predict(crops[i:i + step], imgsz=CROP, conf=self.conf,
                                         verbose=False, half=self.half, **kw)
            out += [[(*map(float, b), float(c)) for b, c in
                     zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]
                    for r in results]
        return out


class TorchBackend:
    """.pt weights run straight through PyTorch: half precision on a GPU.

    Ultralytics' predict() prepares every image and builds a Results object per
    crop; for one 320 px crop that overhead is several times the network itself.
    Here the crops go to the network as one tensor and the raw output is decoded
    like OnnxBackend's (same layout: 4 + classes rows per anchor).
    """

    name = "torch"

    def __init__(self, path, conf=DETECT_CONF, device=None):
        import torch
        from ultralytics import YOLO
        self.torch = torch
        if device in (None, ""):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif str(device).isdigit():
            device = f"cuda:{device}"
        self.device = torch.device(device)
        self.half = self.device.type == "cuda"
        model = YOLO(path, task="detect").model.fuse().eval().to(self.device)
        self.model = model.half() if self.half else model.float()
        self.conf = conf
        self._decode = OnnxBackend.postprocess.__get__(self)
        self.detect([np.zeros((CROP, CROP, 3), np.uint8)])      # first call: kernels, cuDNN

    def detect(self, crops):
        if not len(crops):
            return []
        torch = self.torch
        x = torch.from_numpy(OnnxBackend.preprocess(crops)).to(self.device)
        if self.half:
            x = x.half()
        with torch.inference_mode():
            out = self.model(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return self._decode(out.float().cpu().numpy())


class OnnxBackend:
    """An ultralytics ONNX export (output 1 x (4 + classes) x anchors) on onnxruntime."""

    name = "onnx"

    def __init__(self, path, conf=DETECT_CONF, device=None):
        import onnxruntime as ort
        providers = ["CPUExecutionProvider"]
        if device not in ("cpu",) and "CUDAExecutionProvider" in ort.get_available_providers():
            # The default EXHAUSTIVE search times every cuDNN algorithm again for
            # each new input shape: seconds, in the middle of a rally.
            providers.insert(0, ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}))
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(path, opts, providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.fixed_batch = inp.shape[0] if isinstance(inp.shape[0], int) else None
        self.conf = conf
        # On a GPU, batches are padded to a few fixed sizes, all run once here:
        # a new shape costs a long first call, a few padded crops cost nothing.
        self.on_gpu = "CUDAExecutionProvider" in self.session.get_providers()
        if self.on_gpu and not self.fixed_batch:
            for size in GPU_BATCHES:
                self.session.run(None, {self.input_name:
                                        np.zeros((size, 3, CROP, CROP), np.float32)})

    @staticmethod
    def preprocess(crops):
        """N x CROP x CROP x 3 uint8 (BGR order) -> N x 3 x CROP x CROP float32 RGB."""
        x = np.ascontiguousarray(np.stack(crops)[..., ::-1].transpose(0, 3, 1, 2))
        return x.astype(np.float32) / 255.0

    def postprocess(self, out):
        """(N, 4 + nc, A) raw predictions -> per-crop [(x1, y1, x2, y2, conf), ...]."""
        res = []
        for pred in out:
            scores = pred[4:].max(axis=0)
            keep = scores >= self.conf
            if not keep.any():
                res.append([])
                continue
            cx, cy, w, h = pred[:4, keep]
            boxes = np.clip(np.c_[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 0, CROP)
            res.append(_nms(boxes, scores[keep], self.conf))
        return res

    def detect(self, crops):
        if not len(crops):
            return []
        x = self.preprocess(crops)
        if self.on_gpu and not self.fixed_batch:
            n = len(x)
            outs = []
            for i in range(0, n, GPU_BATCHES[-1]):
                chunk = x[i:i + GPU_BATCHES[-1]]
                pad = batch_bucket(len(chunk)) - len(chunk)
                if pad:
                    chunk = np.concatenate([chunk, np.zeros((pad, *chunk.shape[1:]), np.float32)])
                outs.append(self.session.run(None, {self.input_name: chunk})[0][:len(chunk) - pad])
            return self.postprocess(np.concatenate(outs))
        if self.fixed_batch:
            outs = []
            for i in range(0, len(x), self.fixed_batch):
                chunk = x[i:i + self.fixed_batch]
                pad = self.fixed_batch - len(chunk)
                if pad:
                    chunk = np.concatenate([chunk, np.zeros((pad, *chunk.shape[1:]), np.float32)])
                outs.append(self.session.run(None, {self.input_name: chunk})[0][:len(chunk) - pad])
            out = np.concatenate(outs)
        else:
            out = self.session.run(None, {self.input_name: x})[0]
        return self.postprocess(out)


class DiffBlobBackend:
    """No model: ball-sized blobs of the difference channel, scored by contrast.

    It reads the same crops as the network (channel 1 = difference at time t), so
    the online tracker runs end to end without weights; it is weaker than a
    trained model wherever a racket, a shoe or a ball kid moves.
    """

    name = "diff"

    def __init__(self, threshold=60, min_area=6, max_area=260, max_elongation=5.0,
                 diff_gain=2.0):
        self.threshold = threshold      # on the stored (gain-scaled) difference
        self.min_area, self.max_area = min_area, max_area
        self.max_elongation = max_elongation
        self.diff_gain = diff_gain

    def detect(self, crops):
        res = []
        for crop in crops:
            diff = crop[..., 1]
            mask = (diff >= self.threshold).astype(np.uint8)
            n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            found = []
            for x, y, w, h, area in stats[1:n]:
                if not (self.min_area <= area <= self.max_area):
                    continue
                if max(w, h) > self.max_elongation * max(1, min(w, h)):
                    continue
                peak = float(diff[y:y + h, x:x + w].max()) / 255.0
                found.append((float(x), float(y), float(x + w), float(y + h),
                              0.2 + 0.6 * peak))
            res.append(found)
        return res
