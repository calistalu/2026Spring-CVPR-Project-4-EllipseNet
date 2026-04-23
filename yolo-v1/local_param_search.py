import argparse
import itertools
import json
import math
import os
import random
from datetime import datetime

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data import YoloPascalVocDataset
from loss import SumSquaredErrorLoss
from models import YOLOv1ResNet


def parse_args():
    parser = argparse.ArgumentParser(description="Local hyperparameter search (single program, no wandb).")
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Deprecated. Kept for compatibility; this script now always runs all grid combinations.",
    )
    parser.add_argument("--epochs", type=int, default=12, help="Epochs per trial.")
    parser.add_argument("--train-fraction", type=float, default=0.25, help="Fraction of train split used in each trial.")
    parser.add_argument("--val-fraction", type=float, default=0.35, help="Fraction of val split used in each trial.")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--out-dir", type=str, default=None, help="Output folder. Default: results/hparam_search/<timestamp>")
    return parser.parse_args()


def make_subset(dataset, fraction, seed):
    n = len(dataset)
    k = max(1, int(n * fraction))
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    return Subset(dataset, indices[:k])


def build_grid_configs():
    learning_rates = [1e-3, 1e-4, 1e-5]
    batch_sizes = [64, 128]
    l_coords = [2.0, 5.0]
    l_thetas = [2.0, 5.0]
    l_noobjs = [0.1, 0.5]

    configs = []
    for lr, bs, l_coord, l_theta, l_noobj in itertools.product(
        learning_rates,
        batch_sizes,
        l_coords,
        l_thetas,
        l_noobjs,
    ):
        configs.append(
            {
                "learning_rate": lr,
                "batch_size": bs,
                "l_coord": l_coord,
                "l_theta": l_theta,
                "l_noobj": l_noobj,
                "augment": True,
            }
        )
    return configs


def draw_summary_image(sorted_rows, out_path):
    top_rows = sorted_rows[:10]
    width, height = 1400, 760
    img = Image.new("RGB", (width, height), (247, 248, 250))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    draw.text((20, 16), "Hyperparameter Search Results (Top 10 by val_loss)", fill=(20, 22, 28), font=font)

    chart_x0, chart_y0 = 20, 48
    chart_x1, chart_y1 = 960, 360
    draw.rectangle([chart_x0, chart_y0, chart_x1, chart_y1], outline=(180, 185, 192), width=2)
    if top_rows:
        vals = [r["best_val_loss"] for r in top_rows]
        vmin, vmax = min(vals), max(vals)
        span = max(vmax - vmin, 1e-8)
        bar_w = max(18, (chart_x1 - chart_x0 - 40) // len(top_rows))
        for i, row in enumerate(top_rows):
            norm = (row["best_val_loss"] - vmin) / span
            bar_h = int((1.0 - norm) * (chart_y1 - chart_y0 - 32))
            x = chart_x0 + 20 + i * bar_w
            y = chart_y1 - 16 - bar_h
            draw.rectangle([x, y, x + bar_w - 6, chart_y1 - 16], fill=(66, 133, 244), outline=(66, 133, 244))
            draw.text((x, chart_y1 - 14), str(row["trial_id"]), fill=(30, 33, 40), font=font)
        draw.text((chart_x0 + 8, chart_y0 + 8), f"val_loss range: [{vmin:.4f}, {vmax:.4f}]", fill=(60, 65, 74), font=font)

    table_x, table_y = 20, 390
    headers = ["rank", "trial", "val_loss", "train_loss", "lr", "bs", "l_coord", "l_theta", "l_noobj"]
    col_w = [50, 50, 95, 95, 105, 55, 70, 70, 70]
    x = table_x
    for i, h in enumerate(headers):
        draw.text((x, table_y), h, fill=(20, 22, 28), font=font)
        x += col_w[i]

    y = table_y + 20
    for rank, row in enumerate(top_rows, start=1):
        values = [
            str(rank),
            str(row["trial_id"]),
            f'{row["best_val_loss"]:.5f}',
            f'{row["best_train_loss"]:.5f}',
            f'{row["learning_rate"]:.2e}',
            str(row["batch_size"]),
            f'{row["l_coord"]:.1f}',
            f'{row["l_theta"]:.1f}',
            f'{row["l_noobj"]:.1f}',
        ]
        x = table_x
        for i, v in enumerate(values):
            draw.text((x, y), v, fill=(30, 33, 40), font=font)
            x += col_w[i]
        y += 18

    img.save(out_path)


def write_csv(rows, out_path):
    keys = [
        "trial_id",
        "best_train_loss",
        "best_val_loss",
        "learning_rate",
        "batch_size",
        "l_coord",
        "l_theta",
        "l_noobj",
        "epochs",
        "train_fraction",
        "val_fraction",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join([str(row[k]) for k in keys]) + "\n")


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or os.path.join("results", "hparam_search", now)
    os.makedirs(out_dir, exist_ok=True)

    base_train_set = YoloPascalVocDataset("train", normalize=True, augment=False)
    base_val_set = YoloPascalVocDataset("val", normalize=True, augment=False)
    train_subset = make_subset(base_train_set, args.train_fraction, args.seed + 11)
    val_subset = make_subset(base_val_set, args.val_fraction, args.seed + 23)

    print(f"device={device}")
    print(f"train_subset={len(train_subset)} / {len(base_train_set)}")
    print(f"val_subset={len(val_subset)} / {len(base_val_set)}")
    print(f"output_dir={out_dir}")

    trial_configs = build_grid_configs()
    total_trials = len(trial_configs)
    unique_trial_configs = {
        (
            h["learning_rate"],
            h["batch_size"],
            h["l_coord"],
            h["l_theta"],
            h["l_noobj"],
        )
        for h in trial_configs
    }
    if len(unique_trial_configs) != total_trials:
        raise RuntimeError("Duplicate hyperparameter combinations detected in the grid.")

    if args.trials is not None and args.trials != total_trials:
        print(f"[Info] Ignoring deprecated --trials={args.trials}; running full grid with {total_trials} trials.")
    print(f"total_trials={total_trials}")

    rows = []

    for t, h in enumerate(trial_configs, start=1):
        print("\n" + "=" * 72)
        print(f"Trial {t}/{total_trials}: {json.dumps(h)}")

        train_set_trial = YoloPascalVocDataset("train", normalize=True, augment=True)
        train_subset_trial = Subset(train_set_trial, train_subset.indices)

        train_loader = DataLoader(
            train_subset_trial,
            batch_size=h["batch_size"],
            shuffle=True,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=h["batch_size"],
            shuffle=False,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
            drop_last=False,
        )

        model = YOLOv1ResNet().to(device)
        loss_fn = SumSquaredErrorLoss(l_coord=h["l_coord"], l_theta=h["l_theta"], l_noobj=h["l_noobj"])
        optimizer = torch.optim.Adam(model.parameters(), lr=h["learning_rate"])

        best_train = math.inf
        best_val = math.inf

        for epoch in range(args.epochs):
            model.train()
            train_loss = 0.0
            for data, labels, _ in tqdm(train_loader, desc=f"Trial{t} Train E{epoch+1}", leave=False):
                data = data.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                pred = model(data)
                loss = loss_fn(pred, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() / max(len(train_loader), 1)

            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for data, labels, _ in tqdm(val_loader, desc=f"Trial{t} Val E{epoch+1}", leave=False):
                    data = data.to(device)
                    labels = labels.to(device)
                    pred = model(data)
                    loss = loss_fn(pred, labels)
                    val_loss += loss.item() / max(len(val_loader), 1)

            best_train = min(best_train, train_loss)
            best_val = min(best_val, val_loss)
            print(f"Trial {t} Epoch {epoch+1}/{args.epochs}: train={train_loss:.6f} val={val_loss:.6f}")

        row = {
            "trial_id": t,
            "best_train_loss": float(best_train),
            "best_val_loss": float(best_val),
            "learning_rate": float(h["learning_rate"]),
            "batch_size": int(h["batch_size"]),
            "l_coord": float(h["l_coord"]),
            "l_theta": float(h["l_theta"]),
            "l_noobj": float(h["l_noobj"]),
            "epochs": int(args.epochs),
            "train_fraction": float(args.train_fraction),
            "val_fraction": float(args.val_fraction),
        }
        rows.append(row)

    rows_sorted = sorted(rows, key=lambda r: r["best_val_loss"])
    write_csv(rows_sorted, os.path.join(out_dir, "trials.csv"))
    draw_summary_image(rows_sorted, os.path.join(out_dir, "summary.png"))

    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("Top 5 trials by best_val_loss\n")
        for i, r in enumerate(rows_sorted[:5], start=1):
            f.write(
                f"{i}. trial={r['trial_id']} val={r['best_val_loss']:.6f} train={r['best_train_loss']:.6f} "
                f"lr={r['learning_rate']:.2e} bs={r['batch_size']} "
                f"l_coord={r['l_coord']:.1f} l_theta={r['l_theta']:.1f} l_noobj={r['l_noobj']:.1f}\n"
            )

    best = rows_sorted[0]
    print("\n" + "#" * 72)
    print("Search completed.")
    print(f"Best trial: {best['trial_id']}")
    print(
        f"best_val_loss={best['best_val_loss']:.6f}, best_train_loss={best['best_train_loss']:.6f}, "
        f"lr={best['learning_rate']:.2e}, bs={best['batch_size']}, "
        f"l_coord={best['l_coord']:.1f}, l_theta={best['l_theta']:.1f}, l_noobj={best['l_noobj']:.1f}"
    )
    print(f"Artifacts: {out_dir}")


if __name__ == "__main__":
    main()
