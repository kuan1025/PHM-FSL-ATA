import os, time, re, glob, random, math
import numpy as np
import torch
import torch.optim

from methods.backbone import model_dict
from data.datamgr import SetDataManager
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet, RelationNetLRP
from methods.protonet import ProtoNet
from methods.gnnnet import GnnNet, GnnNetLRP
from methods.tpn import TPN
from options import parse_args

# ---------------- device ----------------
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

device = get_device()
print(f"[Device] Using {device}")

# ---------------- ckpt utils ----------------
def _atomic_save_ckpt(path, epoch, model, optimizer=None, metric=None):
    state_cpu = {}
    for k, v in model.state_dict().items():
        if isinstance(v, torch.Tensor):
            state_cpu[k] = v.detach().to('cpu')
        else:
            state_cpu[k] = v
    payload = {'epoch': int(epoch), 'state': state_cpu}
    if optimizer is not None:
        payload['optimizer'] = optimizer.state_dict()
    if metric is not None and not (isinstance(metric, float) and math.isnan(metric)):
        payload['metric'] = float(metric)
    tmp = path + '.tmp'
    torch.save(payload, tmp)
    os.replace(tmp, path)

def _load_full_ckpt(path, maploc):
    try:
        return torch.load(path, map_location=maploc, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=maploc)

def _auto_resume_ckpt(params):
    ckpt_dir = getattr(params, 'checkpoint_dir', None)
    if not ckpt_dir or not os.path.isdir(ckpt_dir):
        return None

    # explicit epoch
    if getattr(params, 'resume_epoch', 0) > 0:
        p = os.path.join(ckpt_dir, f'{params.resume_epoch}.tar')
        return p if os.path.exists(p) else None

    cand = []
    latest = os.path.join(ckpt_dir, 'latest.tar')
    if os.path.exists(latest):
        try:
            e = _load_full_ckpt(latest, 'cpu').get('epoch', -1)
            cand.append((e, latest))
        except Exception:
            pass
    for p in glob.glob(os.path.join(ckpt_dir, '[0-9]*.tar')):
        m = re.search(r'(\d+)\.tar$', os.path.basename(p))
        if not m:
            continue
        try:
            e = int(m.group(1))
            cand.append((e, p))
        except Exception:
            pass
    if not cand:
        return None
    cand.sort(key=lambda t: t[0])
    return cand[-1][1]

# ---------------- eval helpers ----------------
def ci_halfwidth(p, N):
    """ 95 cli """
    p = max(1e-6, min(1 - 1e-6, p))
    N = max(1, int(N))
    return 1.96 * math.sqrt(p * (1 - p) / N) * 100.0

@torch.no_grad()
def eval_acc(model, loader):
    if loader is None:
        return float('nan')
    was_training = model.training
    model.eval()
    try:
        acc = model.test_loop(loader)
    finally:
        if was_training:
            model.train()
    return float(acc)

def _parse_mix(s: str):
    """'PV:0.7,PD:0.3' -> normalized dict; empty => {}"""
    if not s:
        return {}
    parts = [p.strip() for p in s.split(',') if p.strip()]
    kv = {}
    for p in parts:
        if ':' not in p:
            continue
        k, v = p.split(':', 1)
        try:
            kv[k.strip().upper()] = float(v)
        except ValueError:
            pass
    z = sum(kv.values()) or 1.0
    for k in list(kv.keys()):
        kv[k] = kv[k] / z
    return kv

def _resolve_json(data_dir, dataset, testset, fname):
    if not fname:
        return None
    if os.path.isabs(fname) and os.path.exists(fname):
        return fname
    for p in (
        os.path.join(data_dir, dataset, fname),
        os.path.join(data_dir, testset or dataset, fname),
        os.path.join(data_dir, fname),
        fname,
    ):
        if os.path.exists(p):
            return p
    return None

# ---------------- training (no ATA, no early-stop) ----------------
def train(base_loader, pv_val_loader, pd_val_loader, model, optimizer,
          start_epoch, stop_epoch, params):
    total_it, epoch_times = 0, []
    best_pv = -1.0
    best_pd = -1.0
    best_mix = -1.0

    mix_w = _parse_mix(getattr(params, 'mix_eval', ''))
    eval_every = max(1, int(getattr(params, 'eval_every', 1)))
    val_eps = max(1, int(getattr(params, 'val_episodes', 500)))
    save_freq = max(1, int(getattr(params, 'save_freq', 50)))

    for epoch in range(start_epoch, stop_epoch):
        t0 = time.time()

        # ---- Train（episode-based，model.train_loop）
        model.train()
        total_it = model.train_loop(epoch, base_loader, optimizer, total_it)

        # ---- Eval
        acc_pv = float('nan')
        acc_pd = float('nan')
        acc_mix = float('nan')

        if ((epoch + 1) % eval_every) == 0:
            acc_pv = eval_acc(model, pv_val_loader)
            acc_pd = eval_acc(model, pd_val_loader)

            if mix_w:
                apv = 0.0 if math.isnan(acc_pv) else acc_pv
                apd = 0.0 if math.isnan(acc_pd) else acc_pd
                acc_mix = mix_w.get('PV', 0.0) * apv + mix_w.get('PD', 0.0) * apd

            # update best checkpoints
            if not math.isnan(acc_pv) and acc_pv > best_pv:
                best_pv = acc_pv
                _atomic_save_ckpt(
                    os.path.join(params.checkpoint_dir, 'best_pv.tar'),
                    epoch, model, optimizer, best_pv
                )
            if not math.isnan(acc_pd) and acc_pd > best_pd:
                best_pd = acc_pd
                _atomic_save_ckpt(
                    os.path.join(params.checkpoint_dir, 'best_pd.tar'),
                    epoch, model, optimizer, best_pd
                )
            if not math.isnan(acc_mix) and acc_mix > best_mix:
                best_mix = acc_mix
                _atomic_save_ckpt(
                    os.path.join(params.checkpoint_dir, 'best_mix.tar'),
                    epoch, model, optimizer, best_mix
                )

        # ---- Save periodic & latest
        if ((epoch + 1) % save_freq == 0) or (epoch == stop_epoch - 1):
            ep_path = os.path.join(params.checkpoint_dir, f'{epoch}.tar')
            _atomic_save_ckpt(ep_path, epoch, model, optimizer, None)
        _atomic_save_ckpt(os.path.join(params.checkpoint_dir, 'latest.tar'),
                          epoch, model, optimizer, None)

        # ---- Logs
        ci_pv = ci_halfwidth((acc_pv / 100.0), val_eps) if not math.isnan(acc_pv) else 0.0
        ci_pd = ci_halfwidth((acc_pd / 100.0), val_eps) if not math.isnan(acc_pd) else 0.0
        mix_msg = f" | MIX={acc_mix:.2f} (best={best_mix:.2f})" if not math.isnan(acc_mix) else ""

        ep_time = time.time() - t0
        epoch_times.append(ep_time)
        avg_t = sum(epoch_times) / len(epoch_times)
        eta_min = (stop_epoch - epoch - 1) * avg_t / 60.0

        pv_msg = f"{acc_pv:.2f}±{ci_pv:.2f}" if not math.isnan(acc_pv) else "nan"
        pd_msg = f"{acc_pd:.2f}±{ci_pd:.2f}" if not math.isnan(acc_pd) else "nan"
        print(f"[Epoch {epoch+1}/{stop_epoch}] PV={pv_msg} | PD={pd_msg}{mix_msg} | "
              f"time={ep_time:.1f}s | ETA≈{eta_min:.1f} min")

    return model

# ---------------- main ----------------
if __name__ == '__main__':
    # seeds
    np.random.seed(10); torch.manual_seed(10); random.seed(10)

    params = parse_args()
    print('--- Training ---\n'); print(params)

    # dirs
    params.save_dir = getattr(params, 'save_dir', 'output')
    params.name = getattr(params, 'name', 'run')
    params.checkpoint_dir = f'{params.save_dir}/checkpoints/{params.name}'
    os.makedirs(params.checkpoint_dir, exist_ok=True)

    # datasets / files
    params.data_dir  = getattr(params, 'data_dir', '.')
    params.dataset   = getattr(params, 'dataset', '')
    params.testset   = getattr(params, 'testset', params.dataset)

    base_file_name   = getattr(params, 'train_file', 'base.json')
    pv_val_file_name = getattr(params, 'pv_val_file', 'val.json')
    pd_val_file_name = getattr(params, 'pd_val_file', None)

    base_file   = _resolve_json(params.data_dir, params.dataset, params.testset, base_file_name)
    pv_val_file = _resolve_json(params.data_dir, params.dataset, params.testset, pv_val_file_name)
    pd_val_file = _resolve_json(params.data_dir, params.dataset, params.testset, pd_val_file_name)

    print('\n--- Prepare dataloaders ---')
    print(f'\ttrain: {base_file}')
    print(f'\tPVval: {pv_val_file}')
    print(f'\tPDval: {pd_val_file}' if pd_val_file else '\tPDval: (none)')

    if not base_file or not os.path.exists(base_file):
        raise FileNotFoundError(f"Train file not found: {base_file!r}")
    if pv_val_file and (not os.path.exists(pv_val_file)):
        print(f"[WARN] PV val file not found: {pv_val_file!r} => skip PV validation")
        pv_val_file = None
    if pd_val_file and (not os.path.exists(pd_val_file)):
        print(f"[WARN] PD val file not found: {pd_val_file!r} => skip PD validation")
        pd_val_file = None

    image_size = int(getattr(params, 'image_size', 224))

    # n_way / n_query / n_shot 
    train_n_way = int(getattr(params, 'train_n_way', getattr(params, 'n_way', 5)))
    test_n_way  = int(getattr(params, 'test_n_way',  train_n_way))
    n_shot      = int(getattr(params, 'n_shot', 5))
    n_query     = int(getattr(params, 'n_query',
                       max(1, int(16 * max(1, test_n_way) / max(1, train_n_way)))))

    base_dm = SetDataManager(image_size, n_query=n_query, n_way=train_n_way, n_support=n_shot)
    val_dm  = SetDataManager(image_size, n_query=n_query, n_way=test_n_way,  n_support=n_shot)

    base_loader   = base_dm.get_data_loader(base_file, aug=bool(getattr(params, 'train_aug', False)))
    pv_val_loader = val_dm.get_data_loader(pv_val_file, aug=False) if pv_val_file else None
    pd_val_loader = val_dm.get_data_loader(pd_val_file, aug=False) if pd_val_file else None

    # model
    method = getattr(params, 'method', 'GNN')
    backbone_name = getattr(params, 'model', 'ResNet10')

    if method in ['MatchingNet', 'matchingnet']:
        model = MatchingNet(model_dict[backbone_name], n_way=train_n_way, n_support=n_shot)
    elif method in ['RelationNet', 'relationnet']:
        model = RelationNet(model_dict[backbone_name], n_way=train_n_way, n_support=n_shot)
    elif method in ['RelationNetLRP', 'relationnetlrp']:
        model = RelationNetLRP(model_dict[backbone_name], n_way=train_n_way, n_support=n_shot)
    elif method in ['ProtoNet', 'protonet']:
        model = ProtoNet(model_dict[backbone_name], n_way=train_n_way, n_support=n_shot)
    elif method in ['GNN', 'gnn']:
        model = GnnNet(model_dict[backbone_name], n_way=train_n_way, n_support=n_shot)
    elif method in ['GNNLRP', 'gnnlrp']:
        model = GnnNetLRP(model_dict[backbone_name], n_way=train_n_way, n_support=n_shot)
    elif method in ['TPN', 'tpn']:
        model = TPN(model_dict[backbone_name], n_way=train_n_way, n_support=n_shot)
    else:
        raise ValueError(f"Unsupported method: {method}")
    model = model.to(device)


    if pd_val_loader is not None:
        with torch.no_grad():
            was_training = model.training
            model.eval()
            try:
                for x, _ in pd_val_loader:
                    # x shape: (N, S, 3, H, W)
                    N, S, C, H, W = x.size()
                    n_support = n_shot
                    if S <= n_support:
                        break
                    sup = x[:, :n_support]        # (N, S_sup, 3, H, W)
                    qry = x[:, n_support:]        # (N, S_qry, 3, H, W)
                    sup_flat = sup.reshape(N, n_support, -1)
                    qry_flat = qry.reshape(N, S - n_support, -1)
                    overlap = (sup_flat.unsqueeze(2) == qry_flat.unsqueeze(1)).all(dim=-1)
                    if overlap.any():
                        print("[WARN] PD episode has SUPPORT-QUERY overlap (identical images). This may inflate PD acc.")
                    break
            finally:
                if was_training:
                    model.train()


    model.n_query = n_query

    # optimizer
    lr = float(getattr(params, 'lr', 1e-3))
    weight_decay = float(getattr(params, 'weight_decay', 0.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # resume or pretrain
    start_epoch = int(getattr(params, 'start_epoch', 0))
    stop_epoch  = int(getattr(params, 'stop_epoch', 400))

    resume_path = _auto_resume_ckpt(params) if int(getattr(params, 'resume_epoch', 0)) != 0 else None
    if resume_path and os.path.exists(resume_path):
        ckpt = _load_full_ckpt(resume_path, device)
        state = ckpt['state'] if isinstance(ckpt, dict) and 'state' in ckpt else ckpt
        model.load_state_dict(state, strict=False)
        if isinstance(ckpt, dict) and 'optimizer' in ckpt and ckpt['optimizer']:
            try:
                optimizer.load_state_dict(ckpt['optimizer'])
            except Exception:
                print("[WARN] Optimizer state in resume is incompatible; start optimizer fresh.")
        start_epoch = int(ckpt.get('epoch', -1)) + 1
        print(f"\tResume: {resume_path} (start_epoch={start_epoch})")
    else:
        pre_dir  = os.path.join(params.save_dir, 'checkpoints', getattr(params, 'resume_dir', ''))
        pre_ep   = int(getattr(params, 'pretrain_epoch', -1))
        pre_path = os.path.join(pre_dir, f'{pre_ep}.tar') if pre_ep >= 0 else ''
        if pre_ep >= 0 and os.path.exists(pre_path):
            print(f"\tInit from pretrain: {pre_path}")
            ckpt = _load_full_ckpt(pre_path, 'cpu')
            state = ckpt['state'] if isinstance(ckpt, dict) and 'state' in ckpt else ckpt
            model_params = model.state_dict()
            matched = {k: v for k, v in state.items() if k in model_params and model_params[k].shape == v.shape}
            model_params.update(matched)
            model.load_state_dict(model_params, strict=False)
            print(f"\tLoaded {len(matched)} / {len(model_params)} keys")
        else:
            print('[INFO] Pretrain not found; start from random init.')

    # train
    print('\n--- start training ---')
    try:
        _ = train(base_loader, pv_val_loader, pd_val_loader,
                  model, optimizer, start_epoch, stop_epoch, params)
    except KeyboardInterrupt:
        _atomic_save_ckpt(os.path.join(params.checkpoint_dir, 'latest.tar'),
                          start_epoch, model, optimizer, None)
        print("\n[Interrupted] Saved latest checkpoint.")
