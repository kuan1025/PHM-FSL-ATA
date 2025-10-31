import os, sys
import numpy as np
import torch

from methods.backbone import model_dict
from data.datamgr import SetDataManager
from options import parse_args
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.protonet import ProtoNet
from methods.gnnnet import GnnNet
from methods.tpn import TPN
from tools.logit_collector import LogitCollector

# -------------------- extras: parse & strip unknown flags --------------------

EXTRA_SPECS = {
    "--which_ckpt": str,      # best_mix | best_pv | best_pd | best_model | latest | epoch:123 | <filename.tar> | <path/to/file.tar>
    "--iter_num": int,        # episodes to run
    "--pv_test_file": str,    # e.g., val.micro.json
    "--pd_test_file": str,    # e.g., novel.micro.json
    "--pv_weight": float,     # default 0.5
    "--pd_weight": float,     # default 0.5
    "--progress": int,        # 0/1: show progress bar or periodic logs
    "--dump_dir": str,
}

def pop_extras(argv):
    """Extract extra flags from argv and return (extras_dict, cleaned_argv)."""
    extras = {}
    cleaned = [argv[0]]
    i = 1
    while i < len(argv):
        tok = argv[i]
        if tok in EXTRA_SPECS:
            if i + 1 >= len(argv):
                raise SystemExit(f"[ERROR] Missing value for {tok}")
            caster = EXTRA_SPECS[tok]
            extras[tok] = caster(argv[i+1])
            i += 2
        else:
            cleaned.append(tok)
            i += 1
    return extras, cleaned

# -------------------- utilities --------------------

def get_device():
    if torch.cuda.is_available(): return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): return torch.device('mps')
    return torch.device('cpu')

def resolve_ckpt_path(params, which):
    """
    Supports:
      - Aliases: best_mix / best_pv / best_pd / best_model / latest / epoch:123
      - Filenames: latest.tar / 123.tar (searched underneath the run directory)
      - Paths: relative or absolute /path/to/xxx.tar
    """
    base = f"{params.save_dir}/checkpoints/{params.name}"

    # Direct path or filename (return immediately if it exists)
    if which and which.endswith(".tar"):
        cand = which
        if os.path.isabs(cand) and os.path.exists(cand):
            return cand
        cand2 = os.path.join(base, cand)  # treat as filename under run dir
        if os.path.exists(cand2):
            return cand2
        # Otherwise fall back to the alias flow

    trial = []
    if which in (None, "", "auto", "best_mix"): trial += [os.path.join(base, "best_mix.tar")]
    if which == "best_pv":   trial = [os.path.join(base, "best_pv.tar")]
    if which == "best_pd":   trial = [os.path.join(base, "best_pd.tar")]
    if which == "best_model":trial = [os.path.join(base, "best_model.tar")]
    if which == "latest":    trial = [os.path.join(base, "latest.tar")]
    if which and which.startswith("epoch:"):
        e = which.split(":",1)[1]
        trial = [os.path.join(base, f"{int(e)}.tar")]

    # fallbacks
    trial += [
        os.path.join(base, "best_mix.tar"),
        os.path.join(base, "best_pv.tar"),
        os.path.join(base, "best_pd.tar"),
        os.path.join(base, "best_model.tar"),
        os.path.join(base, "latest.tar"),
    ]
    for p in trial:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No checkpoint found under {base}. Tried top candidates: {trial[:5]}")

def build_model(params, n_way, n_support, device):
    if params.method == 'MatchingNet':
        model = MatchingNet(model_dict[params.model], n_way=n_way, n_support=n_support)
    elif params.method == 'RelationNet':
        model = RelationNet(model_dict[params.model], n_way=n_way, n_support=n_support)
    elif params.method == 'ProtoNet':
        model = ProtoNet(model_dict[params.model], n_way=n_way, n_support=n_support)
    elif params.method == 'GNN':
        model = GnnNet(model_dict[params.model], n_way=n_way, n_support=n_support)
    elif params.method == 'TPN':
        model = TPN(model_dict[params.model], n_way=n_way, n_support=n_support)
    else:
        raise ValueError("Please specify a valid method!")
    return model.to(device)

def load_ckpt_into_model(model, ckpt_path, device, allow_partial=True):
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt['state'] if isinstance(ckpt, dict) and 'state' in ckpt else ckpt
    if allow_partial:
        mp = model.state_dict()
        matched = {k: v for k, v in state.items() if k in mp and mp[k].shape == v.shape}
        mp.update(matched)
        model.load_state_dict(mp, strict=False)
    else:
        model.load_state_dict(state, strict=False)
    return model

@torch.no_grad()
def evaluate_loader(model, loader, n_way, n_support, method, device, show_progress=False, collector=None):
    if loader is None:
        return float('nan'), float('nan'), 0
    if method != 'TPN':
        model.eval()

    acc_all = []
    n_tasks = len(loader)

    # Optional progress display
    pbar = None
    if show_progress:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=n_tasks, ncols=80, desc="Testing")
        except Exception:
            pbar = None  # If tqdm is unavailable, fall back to printing every 50 steps

    for it, (x, _) in enumerate(loader, 1):
        x = x.to(device)
        n_query = x.size(1) - n_support
        model.n_query = n_query

        scores = model.set_forward(x)
        _, top1 = scores.data.topk(1, 1, True, True)
        pred = top1.cpu().numpy().reshape(-1)

        yq = np.repeat(range(n_way), n_query)
        acc = (pred == yq).mean() * 100.0
        acc_all.append(acc)

        if pbar is not None:
            pbar.update(1)
        else:
            if show_progress and (it % 50 == 0 or it == n_tasks):
                print(f"[TEST] progress {it}/{n_tasks} acc(avg)={np.mean(acc_all):.2f}%")
        
        
        if collector is not None:
            logits_2d = scores.detach().cpu().numpy().reshape(-1, scores.size(-1))
            labels_1d = np.repeat(np.arange(n_way), n_query).astype(np.int64)
            collector.add_episode(logits_2d, labels_1d)

    if pbar is not None:
        pbar.close()

    acc_all = np.asarray(acc_all)
    mean = float(np.mean(acc_all)) if acc_all.size else float('nan')
    std  = float(np.std(acc_all))  if acc_all.size else float('nan')
    ci   = (1.96 * std / np.sqrt(max(1, n_tasks))) if acc_all.size else float('nan')
    return mean, ci, n_tasks

def make_loader(json_path, n_way, n_support, n_query, iter_num):
    if not json_path or not os.path.exists(json_path):
        print(f"[TEST] skip loader: {json_path} (not found)")
        return None
    # Handle n_episode / n_eposide naming
    try:
        dm = SetDataManager(224, n_query=n_query, n_way=n_way, n_support=n_support, n_episode=iter_num)
    except TypeError:
        dm = SetDataManager(224, n_query=n_query, n_way=n_way, n_support=n_support, n_eposide=iter_num)
    return dm.get_data_loader(json_path, aug=False)

# -------------------- main --------------------

if __name__ == '__main__':
    # Strip custom flags first so options.parse_args() does not raise "unrecognized"
    extras, cleaned_argv = pop_extras(sys.argv)
    sys_argv_backup = sys.argv
    sys.argv = cleaned_argv
    try:
        params = parse_args()
    finally:
        sys.argv = sys_argv_backup

    device = get_device()

    which_ckpt = extras.get("--which_ckpt", "best_mix")
    iter_num   = extras.get("--iter_num", getattr(params, 'val_episodes', 1000))
    pv_test    = extras.get("--pv_test_file", None)
    pd_test    = extras.get("--pd_test_file", None)
    pv_w       = float(extras.get("--pv_weight", 0.5))
    pd_w       = float(extras.get("--pd_weight", 0.5))
    show_progress = bool(int(extras.get("--progress", 1)))
    dump_dir = extras.get("--dump_dir", None)
    collector_pv = LogitCollector() if dump_dir else None
    collector_pd = LogitCollector() if dump_dir else None

    # n_query: prefer params.n_query if provided, otherwise default to 16
    n_query = int(getattr(params, 'n_query', 16))

    print(f"[TEST] device={device} | which_ckpt={which_ckpt} | iter_num={iter_num} | n_query={n_query}")

    ckpt_path = resolve_ckpt_path(params, which_ckpt)
    print(f"[TEST] load ckpt: {ckpt_path}")

    n_way     = int(params.test_n_way)
    n_support = int(params.n_shot)
    model = build_model(params, n_way=n_way, n_support=n_support, device=device)
    model = load_ckpt_into_model(model, ckpt_path, device)

    # Single or paired evaluation sets
    if pv_test or pd_test:
        # Individual roots: PV uses dataset (P20), PD uses testset (P21)
        pv_root = params.dataset
        pd_root = params.testset

        pv_path = None
        if pv_test:
            pv_path = pv_test if os.path.isabs(pv_test) else os.path.join(params.data_dir, pv_root, pv_test)

        pd_path = None
        if pd_test:
            pd_path = pd_test if os.path.isabs(pd_test) else os.path.join(params.data_dir, pd_root, pd_test)

        print(f"[TEST] pv_path={pv_path}")
        print(f"[TEST] pd_path={pd_path}")

        pv_loader = make_loader(pv_path, n_way, n_support, n_query, iter_num)
        pd_loader = make_loader(pd_path, n_way, n_support, n_query, iter_num)

        pv_mean, pv_ci, pv_N = evaluate_loader(model, pv_loader, n_way, n_support, params.method, device, show_progress, collector=collector_pv)
        pd_mean, pd_ci, pd_N = evaluate_loader(model, pd_loader, n_way, n_support, params.method, device, show_progress, collector=collector_pd)

        print(f"[RESULT] PV: {pv_mean:.2f} ± {pv_ci:.2f}  (episodes={pv_N})")
        print(f"[RESULT] PD: {pd_mean:.2f} ± {pd_ci:.2f}  (episodes={pd_N})")

        w_sum = max(1e-9, pv_w + pd_w)
        mix = (pv_w / w_sum) * (0.0 if np.isnan(pv_mean) else pv_mean) + \
              (pd_w / w_sum) * (0.0 if np.isnan(pd_mean) else pd_mean)
        print(f"[RESULT] MIX({pv_w:.2f}/{pd_w:.2f}): {mix:.2f}")
    else:
        # Single evaluation set: choose dataset or testset root by test_from
        root = params.dataset if params.test_from == 'dataset' else params.testset
        test_json = os.path.join(params.data_dir, root, params.test_file)
        print(f"[TEST] test_json={test_json}")
        if not os.path.exists(test_json):
            raise FileNotFoundError(f"test json not found: {test_json}")
        loader = make_loader(test_json, n_way, n_support, n_query, iter_num)
        if loader is None:
            raise FileNotFoundError(f"loader build failed for: {test_json}")
        mean, ci, N = evaluate_loader(model, loader, n_way, n_support, params.method, device, show_progress)
        print(f"[RESULT] {os.path.basename(test_json)}: {mean:.2f} ± {ci:.2f}  (episodes={N})")
    
    if dump_dir:
        if collector_pv and pv_loader is not None:
            collector_pv.save(os.path.join(dump_dir, "pv_val_logits.npz"))
        if collector_pd and pd_loader is not None:
            collector_pd.save(os.path.join(dump_dir, "pd_test_logits.npz"))
