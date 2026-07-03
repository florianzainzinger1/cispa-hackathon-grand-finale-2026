"""rt_embed.py - ArcFace (buffalo_l w600k_r50) face embedder for retrieval.

Runs the recognition model DIRECTLY on already-aligned 64x64 crops (our gallery
and recons are tight face crops), skipping SCRFD detection which fails on 64px.

Input tensors: (N,3,64,64) float in [0,1], RGB. Output: (N,512) L2-normalized.
Preprocessing matches ArcFace: resize->112x112, RGB->BGR, (x*255-127.5)/127.5.
"""
import os
import sys
import numpy as np

PYLIBS = '/p/scratch/training2625/zainzinger1/1337/recon_work/runs/pylibs'
if PYLIBS not in sys.path:
    sys.path.insert(0, PYLIBS)
os.environ.setdefault('INSIGHTFACE_HOME',
                      '/p/scratch/training2625/zainzinger1/1337/recon_work/runs/insightface_home')

import onnxruntime as ort  # noqa: E402


def _model_path():
    home = os.environ['INSIGHTFACE_HOME']
    return os.path.join(home, 'models', 'buffalo_l', 'w600k_r50.onnx')


class ArcFaceEmbedder:
    def __init__(self, device='cpu'):
        providers = ['CPUExecutionProvider']
        if device == 'cuda':
            avail = ort.get_available_providers()
            if 'CUDAExecutionProvider' in avail:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        so = ort.SessionOptions()
        so.intra_op_num_threads = 8
        self.sess = ort.InferenceSession(_model_path(), sess_options=so, providers=providers)
        self.iname = self.sess.get_inputs()[0].name
        # w600k_r50 expects 112x112
        self.size = 112

    def _prep(self, x255_rgb_112):
        # x255_rgb_112: (N,112,112,3) float in [0,255], RGB
        x = x255_rgb_112[..., ::-1]                 # RGB->BGR
        x = (x - 127.5) / 127.5
        x = np.transpose(x, (0, 3, 1, 2)).astype(np.float32)  # NCHW
        return np.ascontiguousarray(x)

    def embed(self, imgs01, batch=256):
        """imgs01: torch or np (N,3,64,64) float [0,1] RGB. Returns np (N,512) L2-norm."""
        import torch
        if torch.is_tensor(imgs01):
            arr = imgs01.detach().cpu().float().numpy()
        else:
            arr = np.asarray(imgs01, dtype=np.float32)
        N = arr.shape[0]
        # NCHW->NHWC, [0,1]->[0,255]
        nhwc = np.transpose(arr, (0, 2, 3, 1)) * 255.0
        # resize each 64->112 via cv2 (bilinear)
        import cv2
        out = np.empty((N, 512), dtype=np.float32)
        for s in range(0, N, batch):
            chunk = nhwc[s:s + batch]
            r = np.empty((chunk.shape[0], self.size, self.size, 3), dtype=np.float32)
            for i in range(chunk.shape[0]):
                r[i] = cv2.resize(chunk[i], (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            blob = self._prep(r)
            emb = self.sess.run(None, {self.iname: blob})[0]
            n = np.linalg.norm(emb, axis=1, keepdims=True)
            n[n == 0] = 1.0
            out[s:s + batch] = emb / n
        return out


if __name__ == '__main__':
    # self-test: same-identity cosine > cross-identity cosine on the gallery
    import torch
    ROOT = '/p/scratch/training2625/zainzinger1/1337/recon_work'
    g = torch.load(os.path.join(ROOT, 'gallery', 'celeba_top200_cc178.pt'), map_location='cpu')
    imgs = g['images'].float() / 255.0      # (12048,3,64,64)
    cid = g['celeb_id'].numpy()
    is_flip = g['is_flip'].numpy()
    # take non-flip subset for a clean id test
    mask = is_flip == 0
    idx = np.where(mask)[0][:2000]
    emb = ArcFaceEmbedder(device=os.environ.get('FL_DEVICE', 'cpu')).embed(imgs[idx])
    cid_s = cid[idx]
    # same-id vs cross-id cosine
    import itertools
    rng = np.random.default_rng(0)
    same, cross = [], []
    for _ in range(3000):
        i, j = rng.integers(0, len(idx), 2)
        c = float(emb[i] @ emb[j])
        (same if cid_s[i] == cid_s[j] else cross).append(c)
    same = np.array(same); cross = np.array(cross)
    print(f"same-id cos: n={len(same)} mean={same.mean():.3f} med={np.median(same):.3f}")
    print(f"cross-id cos: n={len(cross)} mean={cross.mean():.3f} med={np.median(cross):.3f}")
    print(f"separation (same.mean - cross.mean) = {same.mean()-cross.mean():.3f}")
