"""
Prototype: Exact Closed-Form Convolution Inversion (for CNN full-res Models 10 & 12)

Replaces the Adam-based projected gradient descent optimization in `attack_conv_invert.py` 
with an exact algebraic least-squares solver.
Since the feature map (8x64x64) is highly over-determined relative to the input 
image (3x64x64), the system W*x = f can be solved exactly via pseudo-inverse.
This eliminates all Adam optimization artifacts, blurring, and the need for TV priors.

Usage: python exact_conv_invert.py
"""
import os
import torch
import torch.nn.functional as F

import utils as U
import models as M

def build_conv_matrix(weight, stride=1, padding=1, H=64, W=64):
    """
    Builds the explicit block-Toeplitz dense matrix for a Conv2D operator.
    Uses unit basis vectors to construct the linear operator matrix.
    """
    C_out, C_in, kH, kW = weight.shape
    out_H = (H + 2 * padding - kH) // stride + 1
    out_W = (W + 2 * padding - kW) // stride + 1
    
    in_dim = C_in * H * W
    out_dim = C_out * out_H * out_W
    
    # Pass one-hot vectors through the conv layer to build the matrix
    I = torch.eye(in_dim).view(in_dim, C_in, H, W)
    batch_size = 1024
    
    W_mat_cols = []
    with torch.no_grad():
        for i in range(0, in_dim, batch_size):
            batch = I[i:i+batch_size]
            out = F.conv2d(batch, weight, bias=None, stride=stride, padding=padding)
            W_mat_cols.append(out.view(batch.shape[0], -1))
            
    # W_mat shape: (out_dim, in_dim)
    W_mat = torch.cat(W_mat_cols, dim=0).T
    return W_mat

def exact_invert(name, device="cpu"):
    grad = U.load_grad(name)
    state = U.load_state(name)
    
    if grad.get("family") != "cnn":
        return
        
    meta = {"family": "cnn", "activation": grad["activation"],
            "feature_shape": grad["feature_shape"]}
    model = M.build_model(name, meta, state).to(device)
    
    print(f"\nBuilding exact convolution matrix for {name}...")
    W_mat = build_conv_matrix(model.conv.weight.cpu(), 
                              stride=model.stride, 
                              padding=model.padding).to(device)
    
    feat_path = U.work_path("candidates", name, "analytic_features.pt")
    if not os.path.exists(feat_path):
        print(f"Error: {feat_path} missing. Run attack_analytic.py first.")
        return
        
    fd = torch.load(feat_path, weights_only=False, map_location=device)
    # Target top 128 candidates directly
    feats = fd["feature"][:128]
    
    # 1. Reverse the Activation Function
    act = grad["activation"]
    if act == "relu":
        target = feats.clamp_min(0.0)
    elif act == "sigmoid":
        target = torch.logit(feats.clamp(1e-5, 1 - 1e-5))
    elif act == "tanh":
        target = torch.atanh(feats.clamp(-1 + 1e-5, 1 - 1e-5))
        
    # 2. Subtract the Bias
    bias = model.conv.bias.view(1, -1, 1, 1).to(device)
    target = target - bias
    
    # Flatten targets: (128, out_dim)
    target = target.view(target.shape[0], -1)
    
    print(f"Solving over-determined linear system for {name} (128 images)...")
    # Solve W_mat * X^T = target^T via least squares pseudo-inverse
    # torch.linalg.lstsq solves A @ X = B
    # Our system: W_mat @ X.T = target.T
    X_T, _, _, _ = torch.linalg.lstsq(W_mat, target.T)
    
    # X shape: (128, in_dim)
    X = X_T.T 
    
    # 3. Reshape and clamp to valid image pixel range
    recons = X.view(128, 3, 64, 64).clamp(0, 1).cpu()
    
    out_dir = U.work_path("candidates_exact", name)
    U.ensure_dir(out_dir)
    torch.save({"images": recons, "source": "exact_lstsq"}, 
               os.path.join(out_dir, "exact_candidates.pt"))
    U.save_image_grid(recons[:64], os.path.join(out_dir, "grid_exact.png"))
    print(f"Exact inversion complete -> {out_dir}/grid_exact.png")

if __name__ == "__main__":
    for m in ["model10", "model12"]:
        try:
            exact_invert(m)
        except Exception as e:
            print(f"Failed on {m}: {e}")
