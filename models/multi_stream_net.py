"""Multi-stream forensic network: RGB + ELA + SRM encoders -> gated fusion -> U-Net decoder.

Outputs
  seg_logits : (B,1,H,W)  per-pixel tampering logits  (localisation)
  cls_logits : (B,5)      image class: authentic | copy-move | splicing | removal | retouch

The network is fully convolutional with a global-pooling head, so it accepts any
image size that is a multiple of 8 (see `predict`, which pads automatically).

PyTorch is optional for the project: the classical detector in utils/forensics.py
works without it. If torch is missing this module raises ImportError on import and
models/__init__.py silently falls back.
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.ela import compute_ela
from utils.srm import compute_srm

TORCH_OK = True
INPUT_SIZE = 256
CLASSES = ["authentic", "copy-move", "splicing", "removal", "retouch"]
NUM_CLASSES = len(CLASSES)
CHS_SMALL = (12, 24, 48, 96)
CHS_BASE = (16, 32, 64, 128)


def build_inputs(img_bgr: np.ndarray, size=INPUT_SIZE):
    """BGR uint8 image -> three float tensors (3,H,W): rgb, ela, srm.
    size=None keeps the native resolution (used at inference)."""
    im = img_bgr if size is None else cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - 0.5
    ela = compute_ela(im).astype(np.float32) / 255.0 - 0.5
    srm = compute_srm(im) / 2.0
    to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))
    return to_t(rgb), to_t(ela), to_t(srm)


def _block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class StreamEncoder(nn.Module):
    def __init__(self, in_ch=3, chs=CHS_BASE):
        super().__init__()
        self.blocks = nn.ModuleList()
        c = in_ch
        for o in chs:
            self.blocks.append(_block(c, o))
            c = o

    def forward(self, x):
        feats = []
        for i, b in enumerate(self.blocks):
            x = b(x if i == 0 else F.max_pool2d(x, 2))
            feats.append(x)
        return feats


class GatedFusion(nn.Module):
    """Concatenate the 3 streams, learn per-channel gates (squeeze-excite), project back."""

    def __init__(self, c):
        super().__init__()
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(3 * c, 3 * c // 2, 1), nn.ReLU(inplace=True),
                                  nn.Conv2d(3 * c // 2, 3 * c, 1), nn.Sigmoid())
        self.proj = nn.Sequential(nn.Conv2d(3 * c, c, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True))

    def forward(self, a, b, c):
        x = torch.cat([a, b, c], 1)
        return self.proj(x * self.gate(x))


class MultiStreamForensicNet(nn.Module):
    def __init__(self, chs=CHS_BASE, num_classes=NUM_CLASSES):
        super().__init__()
        chs = tuple(chs)
        self.rgb, self.ela, self.srm = StreamEncoder(3, chs), StreamEncoder(3, chs), StreamEncoder(3, chs)
        self.fuse = nn.ModuleList([GatedFusion(c) for c in chs])
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(len(chs) - 1, 0, -1):
            self.up.append(nn.ConvTranspose2d(chs[i], chs[i - 1], 2, stride=2))
            self.dec.append(_block(2 * chs[i - 1], chs[i - 1]))
        self.seg_head = nn.Conv2d(chs[0], 1, 1)
        self.cls_head = nn.Sequential(nn.Linear(2 * chs[-1] + 1, 64), nn.ReLU(inplace=True),
                                      nn.Dropout(0.3), nn.Linear(64, num_classes))

    def forward(self, rgb, ela, srm):
        fr, fe, fs = self.rgb(rgb), self.ela(ela), self.srm(srm)
        fused = [f(a, b, c) for f, a, b, c in zip(self.fuse, fr, fe, fs)]
        x = fused[-1]
        for up, dec, skip in zip(self.up, self.dec, reversed(fused[:-1])):
            x = dec(torch.cat([up(x), skip], 1))
        seg = self.seg_head(x)
        deep = fused[-1]
        pooled = torch.cat([deep.mean((2, 3)), deep.amax((2, 3)), seg.amax((2, 3))], 1)
        return seg, self.cls_head(pooled)


def load_model(path, device="cpu"):
    try:
        ck = torch.load(path, map_location=device, weights_only=True)
    except TypeError:                                   # older torch without weights_only
        ck = torch.load(path, map_location=device)
    if ck.get("num_classes") != NUM_CLASSES:
        raise ValueError("These weights come from an older, copy-move-only model. "
                         "Delete weights/forensic_net.pt and run:  python force_train.py --preset quick")
    net = MultiStreamForensicNet(tuple(ck.get("chs", CHS_BASE)), NUM_CLASSES)
    net.load_state_dict(ck["state_dict"])
    return net.to(device).eval()


@torch.no_grad()
def predict(net, img_bgr, device="cpu", max_side=640):
    """Returns (class_probabilities float32[5], heat-map float32 HxW in [0,1])."""
    h, w = img_bgr.shape[:2]
    s = min(1.0, max_side / max(h, w))
    im = cv2.resize(img_bgr, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else img_bgr
    ih, iw = im.shape[:2]
    padded = cv2.copyMakeBorder(im, 0, (-ih) % 8, 0, (-iw) % 8, cv2.BORDER_REFLECT)
    rgb, ela, srm = [t.unsqueeze(0).to(device) for t in build_inputs(padded, None)]
    seg, cls = net(rgb, ela, srm)
    probs = torch.softmax(cls, 1)[0].float().cpu().numpy()
    heat = torch.sigmoid(seg)[0, 0].float().cpu().numpy()[:ih, :iw]
    return probs, cv2.resize(heat, (w, h), interpolation=cv2.INTER_LINEAR)
