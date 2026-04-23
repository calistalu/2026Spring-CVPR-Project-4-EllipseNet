import torch
import os
import numpy as np
import argparse
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
import config
from data import YoloPascalVocDataset
from loss import SumSquaredErrorLoss
from models import YOLOv1ResNet

try:
    import wandb
except Exception:
    wandb = None


def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLOv1 on ellipse dataset.')
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=config.EPOCHS)
    parser.add_argument('--learning-rate', type=float, default=config.LEARNING_RATE)
    parser.add_argument('--l-coord', type=float, default=5.0)
    parser.add_argument('--l-theta', type=float, default=5.0)
    parser.add_argument('--l-noobj', type=float, default=0.5)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--scheduler', type=str, choices=['none', 'cosine'], default='none')
    parser.add_argument('--min-lr', type=float, default=1e-6, help='Minimum LR for cosine annealing.')
    parser.add_argument(
        '--unfreeze-backbone-epoch',
        type=int,
        default=-1,
        help='Backbone unfreeze epoch index. -1: keep frozen, 0: train backbone from start, 10: freeze first 10 epochs then unfreeze.'
    )
    parser.add_argument(
        '--train-backbone-last2',
        action='store_true',
        help='Train only backbone layer3+layer4 from epoch 0 (no staged unfreeze).'
    )
    return parser.parse_args()


def set_backbone_trainable(model, trainable):
    backbone = model.model[0]
    backbone.requires_grad_(trainable)


def set_backbone_last2_trainable(model, trainable):
    backbone = model.model[0]
    backbone.layer3.requires_grad_(trainable)
    backbone.layer4.requires_grad_(trainable)


def print_backbone_trainable_overview(model):
    backbone = model.model[0]
    print('[Backbone] trainable overview:')
    for name, module in backbone.named_children():
        params = list(module.parameters())
        if not params:
            continue
        trainable = any(p.requires_grad for p in params)
        trainable_params = sum(p.numel() for p in params if p.requires_grad)
        total_params = sum(p.numel() for p in params)
        print(
            f'  - {name:10s} trainable={str(trainable):5s} '
            f'params={trainable_params}/{total_params}'
        )


if __name__ == '__main__':      # Prevent recursive subprocess creation
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.autograd.set_detect_anomaly(True)         # Check for nan loss
    now = datetime.now()

    model = YOLOv1ResNet().to(device)
    if args.train_backbone_last2:
        set_backbone_trainable(model, False)
        set_backbone_last2_trainable(model, True)
        print('[Backbone] training layer3+layer4 from epoch 0 (no staged unfreeze).')
    elif args.unfreeze_backbone_epoch == 0:
        set_backbone_trainable(model, True)
        print('[Backbone] trainable from epoch 0.')
    elif args.unfreeze_backbone_epoch > 0:
        set_backbone_trainable(model, False)
        print(f'[Backbone] frozen for first {args.unfreeze_backbone_epoch} epochs.')
    else:
        print('[Backbone] keep frozen for all epochs.')
    print_backbone_trainable_overview(model)

    loss_function = SumSquaredErrorLoss(
        l_coord=args.l_coord,
        l_theta=args.l_theta,
        l_noobj=args.l_noobj
    )

    # Adam works better
    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=config.LEARNING_RATE,
    #     momentum=0.9,
    #     weight_decay=5E-4
    # )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate
    )
    scheduler = None
    if args.scheduler == 'cosine':
        total_epochs = config.WARMUP_EPOCHS + args.epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(total_epochs, 1),
            eta_min=args.min_lr
        )

    # Learning rate scheduler (NOT NEEDED)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer,
    #     lr_lambda=utils.scheduler_lambda
    # )

    # Load datasets: train + validation
    train_set = YoloPascalVocDataset('train', normalize=True, augment=True)
    val_set = YoloPascalVocDataset('val', normalize=True, augment=False)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
        shuffle=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        drop_last=True
    )

    # Create folders
    root = os.path.join(
        'models',
        'yolo_v1',
        now.strftime('%m_%d_%Y'),
        now.strftime('%H_%M_%S')
    )
    weight_dir = os.path.join(root, 'weights')
    if not os.path.isdir(weight_dir):
        os.makedirs(weight_dir)

    # Metrics
    train_losses = np.empty((2, 0))
    val_losses = np.empty((2, 0))
    train_errors = np.empty((2, 0))
    val_errors = np.empty((2, 0))


    def save_metrics():
        np.save(os.path.join(root, 'train_losses'), train_losses)
        np.save(os.path.join(root, 'val_losses'), val_losses)
        np.save(os.path.join(root, 'test_losses'), val_losses)
        np.save(os.path.join(root, 'train_errors'), train_errors)
        np.save(os.path.join(root, 'val_errors'), val_errors)
        np.save(os.path.join(root, 'test_errors'), val_errors)


    wandb_run = None
    if config.WANDB_ENABLED:
        if wandb is None:
            print('WANDB_ENABLED=True, but wandb is not installed. Continuing without wandb logging.')
        else:
            try:
                wandb_run = wandb.init(
                    project=config.WANDB_PROJECT,
                    entity=config.WANDB_ENTITY,
                    name=args.run_name or config.WANDB_RUN_NAME,
                    mode=config.WANDB_MODE,
                    settings=wandb.Settings(init_timeout=120),
                    config={
                        'batch_size': args.batch_size,
                        'epochs': args.epochs,
                        'learning_rate': args.learning_rate,
                        'scheduler': args.scheduler,
                        'min_lr': args.min_lr,
                        'unfreeze_backbone_epoch': args.unfreeze_backbone_epoch,
                        'train_backbone_last2': args.train_backbone_last2,
                        'l_coord': args.l_coord,
                        'l_theta': args.l_theta,
                        'l_noobj': args.l_noobj,
                        'dataset': config.ELLIPSE_DATASET,
                        'train_split': config.TRAIN_SPLIT,
                        'val_split': config.VAL_SPLIT,
                        'test_split': config.TEST_SPLIT,
                        'model': 'YOLOv1ResNet',
                        'bbox_attrs': config.BBOX_ATTRS,
                        'ellipse_iou_samples': config.ELLIPSE_IOU_SAMPLES,
                    }
                )
            except Exception as error:
                print(f'wandb.init failed ({error}). Continuing without wandb logging.')
                wandb_run = None


    #####################
    #       Train       #
    #####################
    total_epochs = config.WARMUP_EPOCHS + args.epochs
    for epoch in tqdm(range(total_epochs), desc='Epoch'):
        if (not args.train_backbone_last2) and args.unfreeze_backbone_epoch > 0 and epoch == args.unfreeze_backbone_epoch:
            set_backbone_trainable(model, True)
            print(f'[Backbone] unfrozen at epoch {epoch}.')

        model.train()
        train_loss = 0
        for step, (data, labels, _) in enumerate(tqdm(train_loader, desc='Train', leave=False), start=1):
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            predictions = model.forward(data)
            loss = loss_function(predictions, labels)
            loss.backward()
            optimizer.step()

            tqdm.write(f'[Train] epoch={epoch} step={step}/{len(train_loader)} loss={loss.item():.6f}')

            train_loss += loss.item() / len(train_loader)
            del data, labels

        train_losses = np.append(train_losses, [[epoch], [train_loss]], axis=1)

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for data, labels, _ in tqdm(val_loader, desc='Val', leave=False):
                data = data.to(device)
                labels = labels.to(device)

                predictions = model.forward(data)
                loss = loss_function(predictions, labels)

                val_loss += loss.item() / len(val_loader)
                del data, labels
        val_losses = np.append(val_losses, [[epoch], [val_loss]], axis=1)

        if wandb_run is not None:
            wandb.log({
                'epoch': epoch,
                'loss/train': train_loss,
                'loss/val': val_loss,
                'lr': optimizer.param_groups[0]['lr'],
            }, step=epoch)

        if scheduler is not None:
            scheduler.step()

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(weight_dir, f'epoch_{epoch + 1:03d}.pt'))
            torch.save(model.state_dict(), os.path.join(weight_dir, 'latest.pt'))

        save_metrics()
    save_metrics()
    torch.save(model.state_dict(), os.path.join(weight_dir, 'final'))

    if wandb_run is not None:
        wandb.finish()
