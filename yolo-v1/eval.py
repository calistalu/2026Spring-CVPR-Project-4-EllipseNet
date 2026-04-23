import argparse
import os
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from utils import get_ellipse_iou, get_overlap
from data import YoloPascalVocDataset
from loss import SumSquaredErrorLoss
from models import YOLOv1ResNet

try:
    import wandb
except Exception:
    wandb = None


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate YOLOv1 (repo-style decoding + class-wise NMS) with ellipse-based metrics.'
    )
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights file.')
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--iou-thresholds', type=str, default='0.5', help='Comma separated, e.g. "0.5,0.75".')
    parser.add_argument('--score-threshold', type=float, default=0.2, help='Filter low-confidence detections.')
    parser.add_argument('--max-overlap', type=float, default=0.5, help='Class-wise NMS overlap threshold.')
    parser.add_argument('--save-pr-curve', action='store_true', help='Save PR curve plot(s) to local files.')
    parser.add_argument('--plots-dir', type=str, default=None, help='Directory for saved PR plots/metrics.')
    parser.add_argument('--wandb', action='store_true', help='Log summary metrics to wandb.')
    return parser.parse_args()


def decode_predictions_repo(pred, score_threshold):
    """Repo-style decoding from utils.plot_boxes:
    1) class per cell = argmax(class_probs)
    2) score = class_prob * box_conf
    3) keep both boxes if above threshold
    """
    results = []
    grid_w = 1.0 / config.S
    grid_h = 1.0 / config.S
    for r in range(config.S):
        for c in range(config.S):
            class_probs = pred[r, c, :config.C]
            class_id = int(torch.argmax(class_probs).item())
            class_prob = float(class_probs[class_id].item())
            for b in range(config.B):
                start = config.C + b * config.BBOX_ATTRS
                cx = float(pred[r, c, start + 0].item()) * grid_w + c * grid_w
                cy = float(pred[r, c, start + 1].item()) * grid_h + r * grid_h
                w = float(pred[r, c, start + 2].item())
                h = float(pred[r, c, start + 3].item())
                th = float(pred[r, c, start + 4].item())
                conf = float(pred[r, c, start + 5].item())
                score = class_prob * conf
                if score >= score_threshold:
                    results.append([cx, cy, w, h, th, score, class_id])
    return results


def classwise_nms_repo(boxes, max_overlap):
    """Repo-style NMS using utils.get_overlap and class-wise suppression."""
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda x: x[5], reverse=True)
    num_boxes = len(boxes)
    overlaps = [[0.0 for _ in range(num_boxes)] for _ in range(num_boxes)]
    for i in range(num_boxes):
        for j in range(num_boxes):
            overlaps[i][j] = get_overlap(boxes[i], boxes[j])

    discarded = set()
    kept = []
    for i in range(num_boxes):
        if i in discarded:
            continue

        cx, cy, w, h, th, score, class_id = boxes[i]
        kept.append({
            'class_id': int(class_id),
            'score': float(score),
            'box': (float(cx), float(cy), float(w), float(h), float(th)),
        })

        for j in range(num_boxes):
            other_class = boxes[j][6]
            if j != i and other_class == class_id and overlaps[i][j] > max_overlap:
                discarded.add(j)

    return kept


def decode_ground_truth(label):
    results = []
    grid_w = 1.0 / config.S
    grid_h = 1.0 / config.S
    seen = set()
    for r in range(config.S):
        for c in range(config.S):
            class_one_hot = label[r, c, :config.C]
            if torch.sum(class_one_hot).item() <= 0:
                continue
            class_id = int(torch.argmax(class_one_hot).item())
            start = config.C
            conf = float(label[r, c, start + 5].item())
            if conf <= 0:
                continue
            cx = float(label[r, c, start + 0].item()) * grid_w + c * grid_w
            cy = float(label[r, c, start + 1].item()) * grid_h + r * grid_h
            w = float(label[r, c, start + 2].item())
            h = float(label[r, c, start + 3].item())
            th = float(label[r, c, start + 4].item())

            key = (class_id, round(cx, 5), round(cy, 5), round(w, 5), round(h, 5), round(th, 5))
            if key in seen:
                continue
            seen.add(key)
            results.append({'class_id': class_id, 'box': (cx, cy, w, h, th)})
    return results


def ellipse_iou_pair_from_utils(a_box, b_box):
    """Adapter only: reuse utils.get_ellipse_iou for a single pair."""
    p = torch.zeros((1, 1, 1, config.C + config.B * config.BBOX_ATTRS), dtype=torch.float32)
    a = torch.zeros_like(p)

    a_vals = torch.tensor(a_box, dtype=torch.float32)
    b_vals = torch.tensor(b_box, dtype=torch.float32)
    for box_idx in range(config.B):
        start = config.C + box_idx * config.BBOX_ATTRS
        p[0, 0, 0, start:start + 5] = a_vals
        a[0, 0, 0, start:start + 5] = b_vals
        p[0, 0, 0, start + 5] = 1.0
        a[0, 0, 0, start + 5] = 1.0

    iou_matrix = get_ellipse_iou(p, a, samples=config.ELLIPSE_IOU_SAMPLES)
    return float(iou_matrix[0, 0, 0, 0, 0].item())


def compute_ap(recalls, precisions):
    if not recalls:
        return 0.0
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def save_pr_curve_plot(pr_curves_by_class, ap_by_class, iou_threshold, out_path):
    present = [c for c, d in pr_curves_by_class.items() if d['recall']]
    plt.figure(figsize=(10, 8))
    for cls in present:
        rec = pr_curves_by_class[cls]['recall']
        prec = pr_curves_by_class[cls]['precision']
        ap = ap_by_class.get(cls, 0.0)
        plt.plot(rec, prec, linewidth=1.5, label=f'cls{cls} AP={ap:.3f}')

    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PR Curves (ellipse) @ IoU {iou_threshold:.2f}')
    plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.5)
    if present:
        plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def evaluate_detection(all_preds, all_gts, iou_threshold):
    gt_by_image_class = defaultdict(list)
    gt_count_by_class = defaultdict(int)
    for image_id, gts in all_gts.items():
        for gt in gts:
            cls = gt['class_id']
            gt_by_image_class[(image_id, cls)].append({'box': gt['box'], 'matched': False})
            gt_count_by_class[cls] += 1

    preds_by_class = defaultdict(list)
    for image_id, preds in all_preds.items():
        for p in preds:
            preds_by_class[p['class_id']].append({'image_id': image_id, 'score': p['score'], 'box': p['box']})

    ap_by_class = {}
    precision_by_class = {}
    recall_by_class = {}
    pr_curves_by_class = {}

    total_tp = 0
    total_fp = 0
    total_gt = sum(gt_count_by_class.values())

    for cls in range(config.C):
        preds = sorted(preds_by_class.get(cls, []), key=lambda x: x['score'], reverse=True)
        if gt_count_by_class.get(cls, 0) == 0:
            ap_by_class[cls] = 0.0
            precision_by_class[cls] = 0.0
            recall_by_class[cls] = 0.0
            pr_curves_by_class[cls] = {'recall': [], 'precision': []}
            continue

        tp = []
        fp = []
        for pred in preds:
            image_id = pred['image_id']
            candidates = gt_by_image_class.get((image_id, cls), [])
            best_iou = -1.0
            best_idx = -1
            for idx, gt in enumerate(candidates):
                if gt['matched']:
                    continue
                iou = ellipse_iou_pair_from_utils(pred['box'], gt['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx >= 0 and best_iou >= iou_threshold:
                candidates[best_idx]['matched'] = True
                tp.append(1.0)
                fp.append(0.0)
                total_tp += 1
            else:
                tp.append(0.0)
                fp.append(1.0)
                total_fp += 1

        cum_tp = []
        cum_fp = []
        tps = 0.0
        fps = 0.0
        for i in range(len(tp)):
            tps += tp[i]
            fps += fp[i]
            cum_tp.append(tps)
            cum_fp.append(fps)

        recalls = []
        precisions = []
        gt_n = gt_count_by_class[cls]
        for i in range(len(cum_tp)):
            rec = cum_tp[i] / max(gt_n, 1)
            prec = cum_tp[i] / max(cum_tp[i] + cum_fp[i], 1e-8)
            recalls.append(rec)
            precisions.append(prec)

        ap = compute_ap(recalls, precisions)
        ap_by_class[cls] = ap
        precision_by_class[cls] = precisions[-1] if precisions else 0.0
        recall_by_class[cls] = recalls[-1] if recalls else 0.0
        pr_curves_by_class[cls] = {
            'recall': recalls,
            'precision': precisions,
        }

    present_classes = [c for c in range(config.C) if gt_count_by_class.get(c, 0) > 0]
    mAP = sum(ap_by_class[c] for c in present_classes) / max(len(present_classes), 1)
    precision_micro = total_tp / max(total_tp + total_fp, 1e-8)
    recall_micro = total_tp / max(total_gt, 1e-8)

    return {
        'precision': precision_micro,
        'recall': recall_micro,
        'mAP': mAP,
        'ap_by_class': ap_by_class,
        'precision_by_class': precision_by_class,
        'recall_by_class': recall_by_class,
        'pr_curves_by_class': pr_curves_by_class,
    }


if __name__ == '__main__':
    args = parse_args()
    thresholds = [float(x) for x in args.iou_thresholds.split(',') if x.strip()]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = YOLOv1ResNet().to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()

    loss_function = SumSquaredErrorLoss()
    test_set = YoloPascalVocDataset('test', normalize=True, augment=False)
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        drop_last=False,
        shuffle=False
    )

    all_preds = {}
    all_gts = {}
    image_counter = 0

    with torch.no_grad():
        test_loss = 0.0
        for data, labels, _ in tqdm(test_loader, desc='Test'):
            data = data.to(device)
            labels = labels.to(device)

            predictions = model.forward(data)
            loss = loss_function(predictions, labels)
            test_loss += loss.item() / len(test_loader)

            pred_cpu = predictions.detach().cpu()
            labels_cpu = labels.detach().cpu()
            batch_size = pred_cpu.size(0)
            for i in range(batch_size):
                raw_boxes = decode_predictions_repo(pred_cpu[i], args.score_threshold)
                nms_boxes = classwise_nms_repo(raw_boxes, args.max_overlap)
                gt_boxes = decode_ground_truth(labels_cpu[i])
                all_preds[image_counter] = nms_boxes
                all_gts[image_counter] = gt_boxes
                image_counter += 1
            del data, labels

    print(f'test_split={config.TEST_SPLIT}')
    print(f'test_size={len(test_set)}')
    print(f'test_loss={test_loss:.8f}')
    print(f'iou_thresholds={thresholds}')
    print(f'score_threshold={args.score_threshold}')
    print(f'nms_max_overlap={args.max_overlap}')

    plots_dir = args.plots_dir
    if plots_dir is None:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plots_dir = os.path.join('results', 'eval', stamp)
    os.makedirs(plots_dir, exist_ok=True)

    summary_metrics = {}
    for thr in thresholds:
        metrics = evaluate_detection(all_preds, all_gts, iou_threshold=thr)
        summary_metrics[thr] = metrics
        print(
            f'[IoU@{thr:.2f} ellipse] '
            f'precision={metrics["precision"]:.6f} '
            f'recall={metrics["recall"]:.6f} '
            f'mAP={metrics["mAP"]:.6f}'
        )

        if args.save_pr_curve:
            fig_name = f'pr_curve_ellipse_iou_{thr:.2f}.png'
            fig_path = os.path.join(plots_dir, fig_name)
            save_pr_curve_plot(
                metrics['pr_curves_by_class'],
                metrics['ap_by_class'],
                thr,
                fig_path
            )
            print(f'pr_curve_saved={fig_path}')

    metrics_path = os.path.join(plots_dir, 'eval_metrics.txt')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write(f'weights={os.path.abspath(args.weights)}\n')
        f.write(f'iou_thresholds={thresholds}\n')
        f.write(f'score_threshold={args.score_threshold}\n')
        f.write(f'nms_max_overlap={args.max_overlap}\n')
        f.write(f'test_loss={test_loss:.8f}\n')
        for thr, m in summary_metrics.items():
            f.write(
                f'[IoU@{thr:.2f}] precision={m["precision"]:.6f} '
                f'recall={m["recall"]:.6f} mAP={m["mAP"]:.6f}\n'
            )
    print(f'eval_metrics_saved={metrics_path}')

    if args.wandb:
        if wandb is None:
            print('wandb is not installed; skip wandb logging.')
        else:
            run = wandb.init(
                project=config.WANDB_PROJECT,
                entity=config.WANDB_ENTITY,
                name=(config.WANDB_RUN_NAME + '-eval') if config.WANDB_RUN_NAME else None,
                job_type='eval',
                mode=config.WANDB_MODE if hasattr(config, 'WANDB_MODE') else 'online',
                config={
                    'dataset': config.ELLIPSE_DATASET,
                    'test_split': config.TEST_SPLIT,
                    'weights': os.path.abspath(args.weights),
                    'batch_size': args.batch_size,
                    'num_workers': args.num_workers,
                    'iou_thresholds': thresholds,
                    'score_threshold': args.score_threshold,
                    'nms_max_overlap': args.max_overlap,
                }
            )
            payload = {'loss/test': test_loss}
            for thr, m in summary_metrics.items():
                payload[f'precision@{thr:.2f}'] = m['precision']
                payload[f'recall@{thr:.2f}'] = m['recall']
                payload[f'mAP@{thr:.2f}'] = m['mAP']
            wandb.log(payload)
            run.finish()
