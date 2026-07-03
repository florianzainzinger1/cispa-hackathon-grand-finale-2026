"""Head-leak v2 (agent3): push the board-confirmed ViT head-leak lever (0.3918).
(1) stronger decoder (GroupNorm capacity + multi-scale SSIM loss);
(2) generalize the CLS/penultimate-embedding leak to ANY family -> apply to the
    score-pinning DEFENDED models m6 (CNN) & m8 (MLP): their noisy FIRST layer is
    hopeless, but the classifier HEAD may leak clean per-sample embeddings.
Reuses merge_headleak helpers. GT-free gate = decoder val-SSIM (generalizes to hidden
70% because it adds INFORMATION, not a metric-fit). Non-destructive candidates. GPU.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils as U
import models as M
import vit as VIT
from build_submission import validate_submission
import merge_headleak as HL

DEV = os.environ.get("FL_DEVICE", "cpu")
if DEV == "cuda" and not torch.cuda.is_available():
    DEV = "cpu"


def get_model_and_head(name):
    grad = U.load_grad(name); state = U.load_state(name); fam = grad["family"]
    if fam == "vit":
        model, _, _ = VIT.build_vit(name, state); head = model.head
    elif fam == "mlp":
        model = M.build_model(name, {"family": "mlp", "activation": grad["activation"]}, state)
        head = model.net[5]
    else:  # cnn
        model = M.build_model(name, {"family": "cnn", "activation": grad["activation"],
                                     "feature_shape": grad["feature_shape"]}, state)
        head = model.head
    return model.eval(), head, fam


def feat_extractor(head):
    store = {}
    h = head.register_forward_pre_hook(lambda m, inp: store.__setitem__("f", inp[0].detach()))
    def extract(model, x):
        with torch.no_grad():
            model(x)
        return store["f"]
    return extract, h


class Decoder2(nn.Module):
    def __init__(self, d, ch=192):
        super().__init__()
        self.ch = ch
        self.fc = nn.Linear(d, ch * 4 * 4)
        def blk(i, o):
            return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1), nn.GroupNorm(8, o), nn.GELU())
        self.net = nn.Sequential(blk(ch, ch), blk(ch, ch // 2), blk(ch // 2, ch // 4),
                                 blk(ch // 4, ch // 8), nn.Conv2d(ch // 8, 3, 3, 1, 1), nn.Sigmoid())

    def forward(self, f):
        return self.net(self.fc(f).view(-1, self.ch, 4, 4))


def msssim_loss(a, b, k):
    w = [0.5, 0.3, 0.2]; x, y = a, b; loss = 0.0
    for i in range(3):
        loss = loss + w[i] * (1 - HL.ssim(x, y, k))
        if i < 2:
            x, y = F.avg_pool2d(x, 2), F.avg_pool2d(y, 2)
    return loss


def train_decoder(name, model, head, extract, gallery, d, device, epochs=60, bs=256):
    n = gallery.shape[0]
    perm = torch.randperm(n); nval = min(1024, n // 10)
    val_idx, tr_idx = perm[:nval], perm[nval:]
    feats = []
    with torch.no_grad():
        for i in range(0, n, 512):
            feats.append(extract(model, gallery[i:i + 512].to(device)).cpu())
    F_all = torch.cat(feats, 0)
    D = Decoder2(d).to(device)
    opt = torch.optim.Adam(D.parameters(), lr=1e-3)
    k = HL._gauss_kernel(3, device=device); kb = HL._gauss_kernel(3, win=7, sigma=1.2, device=device)
    best, best_state = -1.0, None
    for ep in range(epochs):
        D.train(); idx = tr_idx[torch.randperm(tr_idx.numel())]
        for i in range(0, idx.numel(), bs):
            bi = idx[i:i + bs]
            f = F_all[bi].to(device); x = gallery[bi].to(device)
            yhat = D(f)
            loss = F.l1_loss(HL.gblur(yhat, kb), HL.gblur(x, kb)) + msssim_loss(yhat, x, k)
            opt.zero_grad(); loss.backward(); opt.step()
        D.eval()
        with torch.no_grad():
            vs = HL.ssim(D(F_all[val_idx].to(device)), gallery[val_idx].to(device), k).item()
        if vs > best:
            best = vs; best_state = {kk: v.detach().clone() for kk, v in D.state_dict().items()}
        if ep % 12 == 0 or ep == epochs - 1:
            print(f"[{name}] ep{ep:3d} val_ssim={vs:.4f} (best={best:.4f})", flush=True)
    D.load_state_dict(best_state)
    return D, best


def run(name, gallery, base, mean, Eig, out_dir):
    model, head, fam = get_model_and_head(name)
    model = model.to(DEV)
    for p in model.parameters():
        p.requires_grad_(False)
    extract, hook = feat_extractor(head)
    d = head.in_features
    embs, info = HL.singleton_embeddings(name)
    ns = len(embs)
    print(f"\n=== {name} [{fam}] d={d} singletons={ns} ===", flush=True)
    D, vss = train_decoder(name, model, head, extract, gallery, d, DEV)
    with torch.no_grad():
        decoded = D(torch.stack([embs[c] for c in sorted(embs)]).to(DEV)).cpu().clamp(0, 1) if ns else torch.zeros(0, 3, 64, 64)
    hook.remove()
    slot = base[name].float().clamp(0, 1).clone(); nfill = 0
    if ns:
        res = HL.face_residual(decoded, mean, Eig)
        ch = decoded.reshape(ns, 3, -1).mean(2)
        imb = ch.max(1).values / (ch.min(1).values + 1e-3)
        thr = float(res.median() + 3 * (res - res.median()).abs().median() * 1.4826)
        good = decoded[~((res > thr) | (imb > 2.6))]
        nfill = min(good.shape[0], 128)
        if nfill:
            slot[:nfill] = good[:nfill]
            U.save_image_grid(good[:64], os.path.join(out_dir, f"hl2_{name}.png"))
    return slot, {"name": name, "family": fam, "d": d, "singletons": ns,
                  "decoder_val_ssim": round(float(vss), 4), "n_filled": nfill}


def main():
    gallery = HL.load_gallery()
    base = torch.load(U.work_path("outputs_final", "submission_current_best.pt"), weights_only=False)
    out_dir = U.work_path("reports", "build", "headleak2"); U.ensure_dir(out_dir)
    mean, Eig = HL.eigenbasis(gallery, k=120)
    names = sys.argv[1:] or ["model2", "model3", "model7", "model10", "model12"]
    report = {}; slots = {}
    for name in names:
        slot, info = run(name, gallery, base, mean, Eig, out_dir)
        slots[name] = slot; report[name] = info
        # save per-model decoded slot so we can compose ANY isolated candidate later
        torch.save({"images": slot.to(torch.float32)}, os.path.join(out_dir, f"slot_{name}.pt"))
        print(f"  {name}: val_ssim={info['decoder_val_ssim']} filled {info['n_filled']}/128", flush=True)
    U.save_json(report, U.work_path("reports", "build", "headleak2_report.json"))
    # convenience candidate = current best with ALL passed models swapped (compose isolated subset later)
    cand = {kk: base[kk].float().clamp(0, 1).clone() for kk in U.MODELS}
    for name in names:
        cand[name] = slots[name]
    if not validate_submission(cand):
        torch.save({kk: cand[kk].to(torch.float32).contiguous() for kk in U.MODELS},
                   U.work_path("outputs_final", "merge_headleak2_batch.pt"))
        print(f"wrote merge_headleak2_batch.pt (current_best + head-leak on {names})")
    print("\nDECIDE per model by decoder_val_ssim + grids; compose isolated candidate from slot_<name>.pt.")


if __name__ == "__main__":
    main()
