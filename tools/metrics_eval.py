import os, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def topk_acc(logits, labels, k=1):
    t = torch.from_numpy(logits)
    y = torch.from_numpy(labels)
    _, pred = t.topk(k, dim=1)
    correct = (pred.eq(y.view(-1,1))).any(dim=1).float().mean().item()
    return correct

def nll_ce(logits, labels, T=1.0):
    t = torch.from_numpy(logits) / float(T)
    y = torch.from_numpy(labels)
    return F.cross_entropy(t, y, reduction='mean').item()

def brier_score(logits, labels, T=1.0):
    t = torch.from_numpy(logits) / float(T)
    p = F.softmax(t, dim=1)
    y = torch.from_numpy(labels)
    onehot = F.one_hot(y, num_classes=p.size(1)).float()
    return torch.mean(torch.sum((p - onehot)**2, dim=1)).item()

def ece_score(logits, labels, T=1.0, n_bins=15):
    t = torch.from_numpy(logits) / float(T)
    p = F.softmax(t, dim=1)
    conf, pred = torch.max(p, dim=1)
    y = torch.from_numpy(labels)
    bins = torch.linspace(0, 1, steps=n_bins+1)
    ece = torch.zeros(1)
    N = p.size(0)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0: 
            continue
        acc_bin = pred[mask].eq(y[mask]).float().mean()
        conf_bin = conf[mask].mean()
        ece += (mask.sum().float()/N) * torch.abs(acc_bin - conf_bin)
    return ece.item()

def reliability_plot(logits, labels, path, T=1.0, n_bins=15):
    t = torch.from_numpy(logits) / float(T)
    p = F.softmax(t, dim=1)
    conf, pred = torch.max(p, dim=1)
    y = torch.from_numpy(labels)
    bins = torch.linspace(0, 1, steps=n_bins+1)
    accs, confs, mids = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0: 
            continue
        accs.append(pred[mask].eq(y[mask]).float().mean().item())
        confs.append(conf[mask].mean().item())
        mids.append(((lo+hi)/2).item())
    plt.figure()
    plt.plot([0,1],[0,1], linestyle='--')
    plt.plot(mids, accs, marker='o')
    plt.plot(mids, confs, marker='x')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy / Confidence')
    plt.title('Reliability diagram')
    plt.tight_layout()
    plt.savefig(path, dpi=160); plt.close()

def risk_coverage(logits, labels, path, T=1.0):
    t = torch.from_numpy(logits) / float(T)
    p = F.softmax(t, dim=1)
    conf, pred = torch.max(p, dim=1)
    y = torch.from_numpy(labels)
    sorted_idx = torch.argsort(conf, descending=True)
    pred = pred[sorted_idx]; y = y[sorted_idx]
    correct = pred.eq(y).float()
    cum = torch.cumsum(correct, dim=0)
    idx = torch.arange(1, len(correct)+1).float()
    coverage = idx / len(correct)
    acc_at_cov = cum / idx
    risk = 1.0 - acc_at_cov
    # AURC（
    aurc = torch.trapz(risk, coverage).item()
    # figure
    plt.figure()
    plt.plot(coverage.numpy(), risk.numpy())
    plt.xlabel('Coverage'); plt.ylabel('Risk (1-acc)')
    plt.title(f'Risk-Coverage (AURC={aurc:.4f})')
    plt.tight_layout()
    plt.savefig(path, dpi=160); plt.close()
    return aurc

def mean_episode_confusion(logits, labels, path, n_way=None, T=1.0):
    t = torch.from_numpy(logits) / float(T)
    probs = F.softmax(t, dim=1)
    _, pred = probs.max(dim=1)
    y = torch.from_numpy(labels)
    if n_way is None: n_way = probs.size(1)
    cm = torch.zeros(n_way, n_way)

    for yi, pi in zip(y, pred):
        cm[yi, pi] += 1
    # normalization
    row_sum = cm.sum(dim=1, keepdim=True).clamp_min(1.0)
    cmn = (cm / row_sum).numpy()
    plt.figure()
    plt.imshow(cmn, vmin=0, vmax=1)
    plt.colorbar()
    plt.xlabel('Pred'); plt.ylabel('True')
    plt.title('Aggregated Confusion (row-normalised)')
    plt.tight_layout()
    plt.savefig(path, dpi=160); plt.close()

def fit_temperature(pv_logits, pv_labels, init_T=1.0, max_iter=50, lr=0.05):
    T = torch.tensor([init_T], requires_grad=True)
    opt = torch.optim.LBFGS([T], lr=lr, max_iter=max_iter, line_search_fn='strong_wolfe')
    pv_logits_t = torch.from_numpy(pv_logits)
    y = torch.from_numpy(pv_labels)
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(pv_logits_t / T.clamp_min(1e-4), y, reduction='mean')
        loss.backward()
        return loss
    opt.step(closure)
    T_val = T.detach().clamp_min(1e-4).item()
    return T_val

def ci95_of_acc(acc_list):
    a = np.array(acc_list, dtype=np.float64)
    m = a.mean()
    s = a.std(ddof=1)
    n = len(a)
    return m, 1.96 * s / np.sqrt(max(n,1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pv', required=True, help='npz: pv_val_logits.npz')
    ap.add_argument('--pd', required=True, help='npz: pd_test_logits.npz')
    ap.add_argument('--outdir', default='./eval_out')
    ap.add_argument('--n_way', type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    pv = np.load(args.pv)
    pd = np.load(args.pd)
    pv_logits, pv_labels = pv['logits'], pv['labels']
    pd_logits, pd_labels = pd['logits'], pd['labels']

    # 1) （T=1）
    base = {
        'Top1': topk_acc(pd_logits, pd_labels, k=1),
        'Top2': topk_acc(pd_logits, pd_labels, k=2),
        'NLL': nll_ce(pd_logits, pd_labels, T=1.0),
        'Brier': brier_score(pd_logits, pd_labels, T=1.0),
        'ECE': ece_score(pd_logits, pd_labels, T=1.0),
    }

    # 2) 
    T = fit_temperature(pv_logits, pv_labels)
    cal = {
        'T': T,
        'Top1': topk_acc(pd_logits/T, pd_labels, k=1),
        'Top2': topk_acc(pd_logits/T, pd_labels, k=2),
        'NLL': nll_ce(pd_logits, pd_labels, T=T),
        'Brier': brier_score(pd_logits, pd_labels, T=T),
        'ECE': ece_score(pd_logits, pd_labels, T=T),
    }

    # 3) 
    reliability_plot(pd_logits, pd_labels, os.path.join(args.outdir, 'reliability_T1.png'), T=1.0)
    reliability_plot(pd_logits, pd_labels, os.path.join(args.outdir, 'reliability_Tcal.png'), T=T)
    aurc1 = risk_coverage(pd_logits, pd_labels, os.path.join(args.outdir, 'risk_coverage_T1.png'), T=1.0)
    aurcT = risk_coverage(pd_logits, pd_labels, os.path.join(args.outdir, 'risk_coverage_Tcal.png'), T=T)
    mean_episode_confusion(pd_logits, pd_labels, os.path.join(args.outdir, 'confusion_Tcal.png'), n_way=args.n_way, T=T)

    # 4) 95% CI
    t = torch.from_numpy(pd_logits / T)
    pred = t.argmax(dim=1).numpy()
    correct01 = (pred == pd_labels).astype(np.float32)
    mean_acc, ci = ci95_of_acc(correct01)

    summary = {
        'PD_unCal': base,
        'PD_TCal': {**cal, 'AURC': aurcT},
        'AURC_unCal': aurc1,
        'Acc_95CI_TCal': [float(mean_acc), float(ci)]
    }
    with open(os.path.join(args.outdir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
