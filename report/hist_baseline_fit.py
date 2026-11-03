"""fit_predict of hist_baseline.py (soft-label MLP), importable without running the script."""
import numpy as np
import torch


def fit_predict(kind, Xtr, Ytr, Xva, Yva, Xte, seed=0):
    torch.manual_seed(seed)
    d = Xtr.shape[1]
    if kind == "linear":
        m = torch.nn.Linear(d, 10)
    else:
        m = torch.nn.Sequential(torch.nn.Linear(d, 256), torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(256, 10))
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr, Ytr, Xva, Yva = map(torch.from_numpy, (Xtr, Ytr, Xva, Yva))
    best, best_state, bad = np.inf, None, 0
    for ep in range(200):
        m.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 512):
            j = perm[i:i + 512]
            loss = -(Ytr[j] * torch.log_softmax(m(Xtr[j]), 1)).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            v = -(Yva * torch.log_softmax(m(Xva), 1)).sum(1).mean().item()
        if v < best - 1e-4:
            best, best_state, bad = v, {k: t.clone() for k, t in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 10:
                break
    m.load_state_dict(best_state)
    m.eval()
    with torch.no_grad():
        return torch.softmax(m(torch.from_numpy(Xte)), 1).numpy().astype(np.float64), best, ep
