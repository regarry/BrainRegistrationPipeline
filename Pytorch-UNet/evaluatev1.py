import torch
import os
import argparse
import logging
import torch.nn.functional as F
from tqdm import tqdm
from utils.data_loading import BasicDataset
from torch.utils.data import DataLoader 
from pathlib import Path
from unet import UNet

from utils.dice_score import multiclass_dice_coeff, dice_coeff
dir_rsc_storage = '/rsstu/users/t/tghashg/MADMbrains/Ryan/Pytorch-UNet/data'
dir_img = Path(dir_rsc_storage+'/cfos_img_labelkit/')
dir_mask = Path(dir_rsc_storage+'/cfos_mask_labelkit/')

@torch.inference_mode()

def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Evaluation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()
    return dice_score / max(num_val_batches, 1)

def get_args():
    parser = argparse.ArgumentParser(description='Evaluate model performance with test data')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images')
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    #parser.add_argument('--viz', '-v', action='store_true',
    #                    help='Visualize the images as they are processed')
    #parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=1,
                        help='Scale factor for the input images')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--cuda', type=int, default=1, help='CUDA device number (int)')
    parser.add_argument('--cpu', type=int, default=8, help='Number of cpu cores (int)')
    parser.add_argument('--nchannels', type=int, default=1, help='Number of channels of input images (int)')
    
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    torch.set_num_threads(args.cpu)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    #in_files = args.input
    #out_files = get_output_filenames(args)

    net = UNet(n_channels=args.nchannels, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}...')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    # Create evaluation dataset and dataloader
    logging.info('Loading Data...')
    dataset = BasicDataset(dir_img, dir_mask, args.scale, mask_suffix='_mask',is_train=False)
    loader_args = dict(batch_size=1, num_workers=os.cpu_count(), pin_memory=True)
    test_loader = DataLoader(dataset, shuffle=False, drop_last=True, **loader_args)
    logging.info('Data Loaded!')

    logging.info('Evaluating...')
    val_score = evaluate(net, test_loader, device, args.amp)
    logging.info(f'Average Dice Score: {val_score}')