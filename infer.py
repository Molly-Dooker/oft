import os
import torch
from torchvision.transforms.functional import to_tensor
from argparse import ArgumentParser
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ipdb
from tqdm import tqdm
from oft import KittiObjectDataset, OftNet, ObjectEncoder, visualize_objects

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('modelpath', type=str,
                        help='path to checkpoint file containing trained model')
    parser.add_argument('-g', '--gpu', type=int, default=0,
                        help='gpu to use for inference (-1 for cpu)')    
    # Data options
    parser.add_argument('--root', type=str, default='data/kitti',
                        help='root directory of the KITTI dataset')
    parser.add_argument('--grid-size', type=float, nargs=2, default=(80., 80.),
                        help='width and depth of validation grid, in meters')
    parser.add_argument('--yoffset', type=float, default=1.74,
                        help='vertical offset of the grid from the camera axis')
    parser.add_argument('--nms-thresh', type=float, default=0.2,
                        help='minimum score for a positive detection')
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
    return parser.parse_args()

def main():
    args = parse_args()
    out_dir = args.modelpath[:-7]
    valid_dir = os.path.join(out_dir,'valid')
    train_dir = os.path.join(out_dir,'train')
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    valid_dataset = KittiObjectDataset(args.root, 'val', args.grid_size, args.grid_res, args.yoffset)
    # 모델 생성 및 로드
    model = OftNet(num_classes=1,
                   frontend=args.frontend,
                   topdown_layers=args.topdown,
                   grid_res=args.grid_res,
                   grid_height=args.grid_height)
    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        model.cuda()
    ckpt = torch.load(args.modelpath, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.eval()
    encoder = ObjectEncoder(nms_thresh=args.nms_thresh)
    fig, (ax_det, ax_gt) = plt.subplots(nrows=2, figsize=(8, 6))
    for idx, image, calib, objects, grid in tqdm(valid_dataset, desc='Validation'):
        image_tensor = to_tensor(image)
        if args.gpu >= 0:
            image_tensor = image_tensor.cuda()
            calib = calib.cuda()
            grid = grid.cuda()
        with torch.no_grad():
            pred_encoded = model(image_tensor[None], calib[None], grid[None])
        pred_encoded = [t[0].cpu() for t in pred_encoded]
        detections = encoder.decode(*pred_encoded, grid.cpu())
        visualize_objects(image_tensor, calib, detections, ax=ax_det)
        ax_det.set_title('Detections')
        visualize_objects(image_tensor, calib, objects, ax=ax_gt)
        ax_gt.set_title('Ground truth')
        save_path = os.path.join(valid_dir, f'frame_{idx:04d}.png')
        fig.savefig(save_path, bbox_inches='tight')
        print(f'Saved {save_path}')
        ax_det.clear()
        ax_gt.clear()
    # train_dataset = KittiObjectDataset(args.root, 'train', args.grid_size, args.grid_res, args.yoffset)
    # for idx, image, calib, objects, grid in tqdm(train_dataset, desc='Train'):
    #     image_tensor = to_tensor(image)
    #     if args.gpu >= 0:
    #         image_tensor = image_tensor.cuda()
    #         calib = calib.cuda()
    #         grid = grid.cuda()
    #     with torch.no_grad():
    #         pred_encoded = model(image_tensor[None], calib[None], grid[None])
    #     pred_encoded = [t[0].cpu() for t in pred_encoded]
    #     detections = encoder.decode(*pred_encoded, grid.cpu())
    #     visualize_objects(image_tensor, calib, detections, ax=ax_det)
    #     ax_det.set_title('Detections')
    #     visualize_objects(image_tensor, calib, objects, ax=ax_gt)
    #     ax_gt.set_title('Ground truth')
    #     save_path = os.path.join(train_dir, f'frame_{idx:04d}.png')
    #     fig.savefig(save_path, bbox_inches='tight')
    #     print(f'Saved {save_path}')
    #     ax_det.clear()
    #     ax_gt.clear()
    plt.close(fig)

if __name__ == '__main__':
    main()