"""Reconstructed model architectures matching the provided state_dicts.

We do not have the original model source, so architectures are inferred from
parameter names/shapes and the gradient bundle's `family`/`activation`/
`feature_shape` metadata, then validated by checking that a forward pass on a
(3,64,64) input reproduces the expected feature/logit shapes.

Only what we need for conv-inversion and gradient-matching is implemented.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


ACTS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "gelu": nn.GELU,
}


def act_module(name):
    return ACTS[name]()


# ----------------------------------------------------------------------------
# MLP  (models 1,4,5,8):  net.0 Linear(12288,1024) - act - net.2 Linear - act -
#                         net.4 (dropout/identity) - net.5 Linear(1024,200)
# ----------------------------------------------------------------------------
class MLPNet(nn.Module):
    def __init__(self, activation="relu", hidden=1024, n_classes=200):
        super().__init__()
        A = lambda: act_module(activation)
        self.net = nn.Sequential(
            nn.Linear(3 * 64 * 64, hidden),   # 0
            A(),                              # 1
            nn.Linear(hidden, hidden),        # 2
            A(),                              # 3
            nn.Identity(),                    # 4 (dropout at train; identity here)
            nn.Linear(hidden, n_classes),     # 5
        )

    def forward(self, x):
        return self.net(x.reshape(x.shape[0], -1))


# ----------------------------------------------------------------------------
# CNN  (models 2,3,6,7,10,12):  conv(3->8,3x3) - act - flatten - fc1 - act - head
# ----------------------------------------------------------------------------
def infer_conv_stride_pad(out_hw, in_hw=64, k=3):
    """Find (stride, padding) giving `out_hw` from `in_hw` with a k x k conv.
    Prefer padding=1 (standard). out = floor((in + 2p - k)/s) + 1."""
    for s in range(1, in_hw + 1):
        for p in (1, 0):
            out = (in_hw + 2 * p - k) // s + 1
            if out == out_hw:
                return s, p
    raise ValueError(f"no stride/pad for {in_hw}->{out_hw}")


class SimpleCNN(nn.Module):
    def __init__(self, feature_shape, activation="relu", hidden=1024, n_classes=200):
        super().__init__()
        C, H, W = feature_shape
        self.feature_shape = tuple(feature_shape)
        self.stride, self.padding = infer_conv_stride_pad(H)
        A = lambda: act_module(activation)
        self.conv = nn.Conv2d(3, C, 3, stride=self.stride, padding=self.padding)
        self.act1 = A()
        self.fc1 = nn.Linear(C * H * W, hidden)
        self.act2 = A()
        self.head = nn.Linear(hidden, n_classes)

    def conv_feature(self, x):
        return self.act1(self.conv(x))

    def forward(self, x):
        f = self.conv_feature(x)
        f = f.reshape(f.shape[0], -1)
        f = self.act2(self.fc1(f))
        return self.head(f)


def build_mlp(state, activation):
    m = MLPNet(activation)
    m.load_state_dict(state)
    return m.eval()


def build_cnn(state, activation, feature_shape):
    m = SimpleCNN(feature_shape, activation)
    m.load_state_dict(state)
    return m.eval()


def build_model(name, meta, state):
    """Return an nn.Module for MLP/CNN families; ViT handled separately."""
    fam = meta["family"]
    act = meta["activation"]
    if fam == "mlp":
        return build_mlp(state, act)
    if fam == "cnn":
        return build_cnn(state, act, meta["feature_shape"])
    raise NotImplementedError(f"family {fam} (vit) not built here")


def validate_forward(model, feature_shape=None):
    """Sanity check: forward on a dummy image yields (1,200) logits and, for
    CNN, the expected conv-feature shape."""
    x = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    ok = out.shape == (2, 200)
    feat_ok = True
    if feature_shape is not None and hasattr(model, "conv_feature"):
        with torch.no_grad():
            f = model.conv_feature(x)
        feat_ok = tuple(f.shape[1:]) == tuple(feature_shape)
    return ok and feat_ok, tuple(out.shape)
