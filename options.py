import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='Few-shot training/eval options (P20 train, eval on P20 & P21; ATA optional)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ---------------- Datasets / files ----------------
    parser.add_argument('--dataset', default='miniImagenet',
                        help='Name or path used with data_dir for training set root')
    parser.add_argument('--testset', default='miniImagenet',
                        help='Name or path used with data_dir for test/target set root')
    parser.add_argument('--data_dir', default='filelists', type=str,
                        help='Root dir that contains dataset folders or JSON lists')

    parser.add_argument('--train_file', type=str, default='base.json',
                        help='Training JSON (relative to data_dir/dataset/ if not absolute)')
    parser.add_argument('--pv_val_file', type=str, default='val.json',
                        help='Source-domain (P20) validation JSON (for selection)')
    parser.add_argument('--pd_val_file', type=str, default=None,
                        help='Optional: Target-domain (P21) JSON for side evaluation only '
                             '(abs path or relative to data_dir/{dataset|testset}).')

    # ---------------- Model / method ----------------
    parser.add_argument('--model', default='ResNet10',
                        help='Backbone name (e.g., ResNet10/ResNet18/ResNet34)')
    parser.add_argument('--method', default='ProtoNet',
                        help='MatchingNet/RelationNet/RelationNetLRP/ProtoNet/GNN/GNNLRP/TPN')

    # ---------------- Episode config ----------------
    parser.add_argument('--train_n_way', default=5, type=int, help='Class count for training episodes')
    parser.add_argument('--test_n_way',  default=5, type=int, help='Class count for validation/test episodes')
    parser.add_argument('--n_shot',      default=5, type=int, help='Supports per class')
    parser.add_argument('--n_query',     default=3, type=int, help='Queries per class in each episode')
    parser.add_argument('--image_size',  default=224, type=int, help='Input image size')
    parser.add_argument('--train_aug',   action='store_true', help='Enable data augmentation during training')

    # ---------------- Optimizer / regularization ----------------
    parser.add_argument('--lr',           default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=0.0,  type=float, help='Weight decay')

    # ---------------- Training schedule / ckpt ----------------
    parser.add_argument('--name',        default='run',   type=str, help='Run name (used in checkpoint path)')
    parser.add_argument('--save_dir',    default='output', type=str, help='Checkpoints root dir')
    parser.add_argument('--save_freq',   default=40, type=int, help='Save every N epochs')
    parser.add_argument('--start_epoch', default=0,  type=int, help='Starting epoch (inclusive)')
    parser.add_argument('--stop_epoch',  default=400, type=int, help='Stopping epoch (exclusive)')
    parser.add_argument('--resume_epoch', default=0, type=int,
                        help='Resume specific epoch from save_dir/checkpoints/name/. '
                             'Set -1 to auto-pick latest; 0 to disable.')

    # ---------------- Pretrain (optional) ----------------
    parser.add_argument('--resume_dir',     default='Pretrain', type=str,
                        help='Directory under save_dir/checkpoints/ for loading pretrain')
    parser.add_argument('--pretrain_epoch', default=399, type=int,
                        help='Which pretrain epoch to load (only if resume_epoch==0 and file exists)')
    parser.add_argument('--num_classes',    default=200, type=int,
                        help='Classifier size used during backbone pretrain (if applicable)')

    # ---------------- ATA knobs (compat / for train_ATA.py) ----------------
    parser.add_argument('--adv_prob', default=0.0, type=float,
                        help='Probability to apply adversarial task augmentation in an episode (0 disables)')
    parser.add_argument('--adv_steps', default=0, type=int,
                        help='Number of adversarial steps per episode (0 disables)')
    # legacy names kept for old scripts (won’t be used by train.py but safe to keep)
    parser.add_argument('--max_lr', default=80., type=float, help='(legacy) max-phase LR')
    parser.add_argument('--T_max',  default=5,   type=int,   help='(legacy) max-phase steps per episode')
    parser.add_argument('--lamb',   default=1.,  type=float, help='(legacy) alpha')
    parser.add_argument('--prob',   default=0.5, type=float, help='(legacy) prob for RCNN originals')

    # ---------------- Evaluation (no early-stop in train.py) ----------------
    parser.add_argument('--eval_every',    type=int, default=1,   help='Evaluate every N epochs')
    parser.add_argument('--val_episodes',  type=int, default=500, help='Episodes per eval (affects CI stability)')
    parser.add_argument('--mix_eval',      type=str, default='',
                        help='Weighted metric like "PV:0.7,PD:0.3". Empty to disable.')

    # ---------------- Legacy test controls (compat) ----------------
    parser.add_argument('--test_from', type=str, default='testset', choices=['dataset','testset'],
                        help='(legacy) root for testing json when needed by old scripts')
    parser.add_argument('--test_file', type=str, default='novel.json',
                        help='(legacy) filename for testing json')

    # ---------------- Misc / reproducibility ----------------
    parser.add_argument('--seed',   default=10, type=int, help='Random seed')
    parser.add_argument('--device', default='', type=str,
                        help='Force device: "cpu"|"cuda"|"mps" (empty = auto detect)')

    # ---------------- Finetune (reserved) ----------------
    parser.add_argument('--finetune_epoch', default=50, type=int, help='Reserved for future use')

    return parser.parse_args()
