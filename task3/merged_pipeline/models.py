"""Model architectures matching the provided state_dicts (ported from teammate's
models.py). Inferred from param names/shapes + gradient metadata, validated by a
forward pass. Only what conv-inversion needs is implemented.
"""
import torch.nn as nn

ACTS = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "gelu": nn.GELU}


def act_module(name):
    return ACTS[name]()


class MLPNet(nn.Module):
    """models 1,4,5,8: net.0 Linear(12288,1024) - act - net.2 - act - Id - net.5."""
    def __init__(self, activation="relu", hidden=1024, n_classes=200):
        super().__init__()
        A = lambda: act_module(activation)
        self.net = nn.Sequential(
            nn.Linear(3 * 64 * 64, hidden), A(),
            nn.Linear(hidden, hidden), A(),
            nn.Identity(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x.reshape(x.shape[0], -1))


def infer_conv_stride_pad(out_hw, in_hw=64, k=3):
    for s in range(1, in_hw + 1):
        for p in (1, 0):
            if (in_hw + 2 * p - k) // s + 1 == out_hw:
                return s, p
    raise ValueError(f"no stride/pad for {in_hw}->{out_hw}")


class SimpleCNN(nn.Module):
    """models 2,3,6,7,10,12: conv(3->C,3x3) - act - flatten - fc1 - act - head."""
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
        f = self.conv_feature(x).reshape(x.shape[0], -1)
        return self.head(self.act2(self.fc1(f)))


def build_cnn(name, meta, state):
    m = SimpleCNN(meta["feature_shape"], meta["activation"])
    m.load_state_dict(state)
    return m.eval()
