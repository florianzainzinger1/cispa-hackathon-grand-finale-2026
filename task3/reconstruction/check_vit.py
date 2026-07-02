"""Validate that the ViT reconstructions load the real state_dicts exactly and
run a forward. Any missing/unexpected keys mean the architecture is wrong."""
import torch
import utils as U
import vit as V

for name in ["model9", "model11"]:
    state = U.load_state(name)
    model, missing, unexpected = V.build_vit(name, state)
    x = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    print(f"{name}: out={tuple(out.shape)} "
          f"missing={list(missing)} unexpected={list(unexpected)}")
    # param-key alignment with the gradient bundle
    grad = U.load_grad(name)
    gkeys = set(grad["gradients"].keys())
    pkeys = set(dict(model.named_parameters()).keys())
    print(f"   grad-keys not in model: {sorted(gkeys - pkeys)[:6]}")
    print(f"   model-keys not in grad: {sorted(pkeys - gkeys)[:6]}")
