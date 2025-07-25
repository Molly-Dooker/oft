import os
from argparse import ArgumentParser
import torch
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from torch import optim
from torch.utils.data import DataLoader
from loguru import logger
import sys
import yaml
import ipdb
import oft
from oft import OftNet, KittiObjectDataset, MetricDict, masked_l1_loss, heatmap_loss, ObjectEncoder, focal_loss
from accelerate import Accelerator
accelerator = Accelerator()

def logger_setup(prefix: str = '', logpath: str = './logs'):
    def console_filter(record):
        return not record["extra"].get("file_only", False)
    def file_filter(record):
        return "console_only" not in record["extra"]
    logger.remove()
    LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {extra[prefix]} | {level} | {message}"
    logger.add(
        sys.stdout,
        level="INFO",
        format=LOG_FORMAT,
        filter=console_filter
    )
    logger.add(
        f"{logpath}/log",
        rotation="500 MB",
        level="INFO",
        format=LOG_FORMAT,
        filter=file_filter
    )
    return logger.bind(prefix=prefix)

def train(args, dataloader, model, encoder, optimizer, epoch):
    if accelerator.is_main_process: logger.bind(console_only=True).info('==> Training on {} minibatches'.format(len(dataloader)))
    model.train()
    epoch_loss = oft.MetricDict()
    for i, (_, image, calib, objects, grid) in enumerate(dataloader):
        pred_encoded = model(image, calib, grid)
        gt_encoded = encoder.encode_batch(objects, grid)
        loss, loss_dict = compute_loss(pred_encoded, gt_encoded, args.loss_weights)
        if torch.isnan(loss): 
            raise RuntimeError('Loss diverged :(')
        gathered_total_loss = accelerator.gather(loss_dict['total'])
        gathered_score_loss = accelerator.gather(loss_dict['score'])
        gathered_pos_loss = accelerator.gather(loss_dict['position'])
        gathered_dim_loss = accelerator.gather(loss_dict['dimension'])
        gathered_ang_loss = accelerator.gather(loss_dict['angle'])
        synced_loss_dict = {
            'total': torch.mean(gathered_total_loss).item(),
            'score': torch.mean(gathered_score_loss).item(),
            'position': torch.mean(gathered_pos_loss).item(),
            'dimension': torch.mean(gathered_dim_loss).item(),
            'angle': torch.mean(gathered_ang_loss).item(),
        }
        epoch_loss += synced_loss_dict
        # Optimize
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        if accelerator.is_main_process:
            # if i % args.print_iter == 0 and i != 0:     
            s = f'[{i:4d}/{len(dataloader):4d}] '
            for k, v in loss_dict.items(): s += f'{k}: {v:.2e} '
            logger.bind(console_only=True).info(s)
    if accelerator.is_main_process:
        logger.info('==> Training complete')
        for key, value in epoch_loss.mean.items():
            logger.info('{:8s}: {:.4e}'.format(key, value))

def validate(args, dataloader, model, encoder, epoch):
    if accelerator.is_main_process: logger.bind(console_only=True).info('==> Validating on {} minibatches'.format(len(dataloader)))
    model.eval()
    epoch_loss = MetricDict()
    for i, (_, image, calib, objects, grid) in enumerate(dataloader):
        with torch.no_grad():
            # Run network forwards
            pred_encoded = model(image, calib, grid)
            # Encode ground truth objects
            gt_encoded = encoder.encode_batch(objects, grid)
            _, loss_dict_tensors = compute_loss(pred_encoded, gt_encoded, args.loss_weights)      
            gathered_total = accelerator.gather(loss_dict_tensors['total'])
            gathered_score = accelerator.gather(loss_dict_tensors['score'])
            gathered_pos = accelerator.gather(loss_dict_tensors['position'])
            gathered_dim = accelerator.gather(loss_dict_tensors['dimension'])
            gathered_ang = accelerator.gather(loss_dict_tensors['angle'])
            synced_loss_dict = {
                'total': torch.mean(gathered_total).item(),
                'score': torch.mean(gathered_score).item(),
                'position': torch.mean(gathered_pos).item(),
                'dimension': torch.mean(gathered_dim).item(),
                'angle': torch.mean(gathered_ang).item(),
            }
            epoch_loss += synced_loss_dict       
    if accelerator.is_main_process:
        logger.info('==> Validation complete')
        for key, value in epoch_loss.mean.items():
            logger.info(f'{key:8s}: {value:.4e}')

def compute_loss(pred_encoded, gt_encoded, loss_weights=[1., 1., 1., 1.]):
    score, pos_offsets, dim_offsets, ang_offsets = pred_encoded
    heatmaps, gt_pos_offsets, gt_dim_offsets, gt_ang_offsets, mask = gt_encoded
    score_weight, pos_weight, dim_weight, ang_weight = loss_weights
    score_loss = heatmap_loss(score, heatmaps)
    pos_loss = masked_l1_loss(pos_offsets, gt_pos_offsets, mask.unsqueeze(2))
    dim_loss = masked_l1_loss(dim_offsets, gt_dim_offsets, mask.unsqueeze(2))
    ang_loss = masked_l1_loss(ang_offsets, gt_ang_offsets, mask.unsqueeze(2))
    score_loss *= score_weight; pos_loss *= pos_weight; dim_loss *= dim_weight; ang_loss *= ang_weight
    total_loss = score_loss + pos_loss + dim_loss + ang_loss
    loss_dict = {
        'score' : score_loss, 'position' : pos_loss,
        'dimension' : dim_loss, 'angle' : ang_loss,
        'total' : total_loss
    }
    return total_loss, loss_dict

def parse_args():
    parser = ArgumentParser()
    # Data options
    parser.add_argument('--root', type=str, default='data/kitti',
                        help='root directory of the KITTI dataset')
    parser.add_argument('--grid-size', type=float, nargs=2, default=(80., 80.),
                        help='width and depth of validation grid, in meters')
    parser.add_argument('--train-grid-size', type=int, nargs=2, 
                        default=(120, 120),
                        help='width and depth of training grid, in pixels')
    parser.add_argument('--grid-jitter', type=float, nargs=3, 
                        default=[.25, .5, .25],
                        help='magn. of random noise applied to grid coords')
    parser.add_argument('-ts','--train-image-size', type=int, nargs=2, 
                        default=(1080, 360), # 1224 370
                        help='size of random image crops during training')
    parser.add_argument('--yoffset', type=float, default=1.74,
                        help='vertical offset of the grid from the camera axis')
    # Model options
    parser.add_argument('--grid-height', type=float, default=4.,
                        help='size of grid cells, in meters')
    parser.add_argument('-r', '--grid-res', type=float, default=0.5,
                        help='size of grid cells, in meters')
    parser.add_argument('--frontend', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34'],
                        help='name of frontend ResNet architecture')
    parser.add_argument('--topdown', type=int, default=8,
                        help='number of residual blocks in topdown network')
    # Optimization options
    parser.add_argument('-l', '--lr', type=float, default=1e-5,
                        help='learning rate')
    parser.add_argument('--warm', type=int, default=10,
                        help='warmup epochs')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for SGD')
    parser.add_argument('-wd', '--weight-decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--lr-decay', type=float, default=0.99,
                        help='factor to decay learning rate by every epoch')
    parser.add_argument('--loss-weights', type=float, nargs=4, 
                        default=[1., 1., 1., 1.],
                        help="loss weighting factors for score, position,"\
                            " dimension and angle loss respectively")
    # Training options
    parser.add_argument('-e', '--epochs', type=int, default=600,
                        help='number of epochs to train for')
    parser.add_argument('-b', '--batch-size', type=int, default=8,
                        help='mini-batch size for training')
    parser.add_argument('-ls', '--lr-scheduler', choices=['cs', 'cswr'], default='cs',
                        help='cs:CosineAnnealing, cswr:CosineAnnealingWarmRestarts')
    parser.add_argument('--cyclestep', type=int, default=100,
                        help='cswr cycle step')
    # Experiment options
    parser.add_argument('name', type=str, default='test',
                        help='name of experiment')
    parser.add_argument('-s', '--savedir', type=str, 
                        default='experiments',
                        help='directory to save experiments to')
    parser.add_argument('-w', '--workers', type=int, default=8,
                        help='number of worker threads to use for data loading')
    parser.add_argument('-vi','--val-interval', type=int, default=5,
                        help='number of epochs between validation runs')
    parser.add_argument('--print-iter', type=int, default=10,
                        help='print loss summary every N iterations')
    return parser.parse_args()

def save_checkpoint(args, epoch, model, optimizer, scheduler):
    unwrapped_model = accelerator.unwrap_model(model)    
    ckpt = {
        'epoch' : epoch,
        'model' : unwrapped_model.state_dict(), # Unwrapped 모델의 state_dict를 저장
        'optim' : optimizer.state_dict(),
        'scheduler' : scheduler.state_dict(),
    }
    ckpt_file = os.path.join(
        args.savedir, args.name, 'checkpoint-{:04d}.pth.gz'.format(epoch))
    logger.bind(console_only=True).info('==> Saving checkpoint \'{}\''.format(ckpt_file))
    torch.save(ckpt, ckpt_file)

def main(args):
    if accelerator.is_main_process: 
        logger.info("===== ACCELERATOR CONFIGURATION =====")
        state_dict = {k: v for k, v in accelerator.state.__dict__.items() if not k.startswith('_')}
        accel_config_str = yaml.dump(state_dict, default_flow_style=False)
        logger.info(f"\n{accel_config_str}")        
        logger.info("===== SCRIPT ARGUMENTS =====")
        args_config_str = yaml.dump(args.__dict__, default_flow_style=False)
        logger.info(f"\n{args_config_str}")
        logger.info("===================================")
    args.lr = args.lr*accelerator.num_processes
    train_data = KittiObjectDataset(
        args.root, 'train', args.grid_size, args.grid_res, args.yoffset)
    val_data = KittiObjectDataset(
        args.root, 'val', args.grid_size, args.grid_res, args.yoffset)
    train_data = oft.AugmentedObjectDataset(
        train_data, args.train_image_size, args.train_grid_size, 
        jitter=args.grid_jitter)
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True, collate_fn=oft.utils.collate, num_workers=args.workers)
    val_loader   = DataLoader(val_data, args.batch_size, shuffle=False, collate_fn=oft.utils.collate, num_workers=args.workers)
    model = OftNet(num_classes=1, frontend=args.frontend, 
                   topdown_layers=args.topdown, grid_res=args.grid_res, 
                   grid_height=args.grid_height)
    encoder = ObjectEncoder()
    # optimizer = optim.SGD(model.parameters(), args.lr, args.momentum, args.weight_decay)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.lr_decay)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.lr_scheduler=='cs':
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warm, eta_min=0)
        warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=args.warm)
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[args.warm])
    elif args.lr_scheduler=='cswr':
        args.cyclestep = args.cyclestep*accelerator.num_processes
        args.warm      = args.warm*accelerator.num_processes
        scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=args.cyclestep, cycle_mult=1.0, max_lr=args.lr, min_lr=1e-12, warmup_steps=args.warm, gamma=0.5)

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, val_loader, scheduler)
    for epoch in range(1, args.epochs+1):
        if accelerator.is_main_process: logger.info(f'=== epoch {epoch} of {args.epochs} ===')
        train(args, train_loader, model, encoder, optimizer, epoch)
        if epoch % args.val_interval == 0:
            optimizer.zero_grad()
            if accelerator.is_main_process: save_checkpoint(args, epoch, model, optimizer, scheduler)
            validate(args, val_loader, model, encoder, epoch)
        scheduler.step()
if __name__ == '__main__':
    args = parse_args()
    logger = logger_setup(prefix=args.name, logpath=os.path.join(args.savedir,args.name))
    main(args)