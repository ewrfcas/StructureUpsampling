from dataset import SketchDataset
from model import *
from utils import *
import argparse
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='test', help='The name of this exp')
    parser.add_argument('--config_file', type=str, default='configs/config.yml')
    parser.add_argument('--GPU_ids', type=str, default='0')
    parser.add_argument('--ckpt_path', type=str, default='./ckpt')

    args = parser.parse_args()
    args.ckpt_path = os.path.join(args.ckpt_path, args.name)
    config_path = os.path.join(args.ckpt_path, 'config.yml')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU_ids
    config = Config(config_path)

    set_seed(42)
    device = torch.device("cuda")
    val_dataset = SketchDataset(config, config.test_flist, augment=False, aug_for_alias=False, training=False)

    val_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True,
                            batch_size=8, num_workers=4)

    model = StructureUpsampling4()
    model = model.to(device)
    model.load_state_dict(torch.load(os.path.join(args.ckpt_path, 'last.pth'))['model'])

    # val_path_line_512 = os.path.join(args.ckpt_path, 'validation_512', 'line')
    # val_path_edge_512 = os.path.join(args.ckpt_path, 'validation_512', 'edge')
    # val_path_line_1024 = os.path.join(args.ckpt_path, 'validation_1024', 'line')
    # val_path_edge_1024 = os.path.join(args.ckpt_path, 'validation_1024', 'edge')
    val_path_line_512 = os.path.join(args.ckpt_path, 'test_512', 'line')
    val_path_edge_512 = os.path.join(args.ckpt_path, 'test_512', 'edge')
    val_path_line_1024 = os.path.join(args.ckpt_path, 'test_1024', 'line')
    val_path_edge_1024 = os.path.join(args.ckpt_path, 'test_1024', 'edge')
    create_dir(val_path_line_512)
    create_dir(val_path_edge_512)
    create_dir(val_path_line_1024)
    create_dir(val_path_edge_1024)

    model.eval()
    eval_losses = []
    with torch.no_grad():
        for items in tqdm(val_loader):
            for k in items:
                if k != 'name':
                    items[k] = items[k].to(device)

            line_out, _ = model.forward(items['line_small'])
            edge_out, _ = model.forward(items['edge_small'])
            line_out = torch.sigmoid(line_out)
            edge_out = torch.sigmoid(edge_out)

            # for i in range(items['line_small'].shape[0]):
            #     line_ = (line_out[i, 0, ...].cpu().numpy() * 255).astype(np.uint8)
            #     edge_ = (edge_out[i, 0, ...].cpu().numpy() * 255).astype(np.uint8)
            #     cv2.imwrite(val_path_line_512 + '/' + items['name'][i], line_)
            #     cv2.imwrite(val_path_edge_512 + '/' + items['name'][i], edge_)

            line_out[line_out > 0.35] = 1
            edge_out[edge_out > 0.35] = 1
            line_out, _ = model.forward(line_out.contiguous())
            edge_out, _ = model.forward(edge_out.contiguous())
            line_out = torch.sigmoid(line_out)
            edge_out = torch.sigmoid(edge_out)
            line_out[line_out > 0.4] = 1
            edge_out[edge_out > 0.4] = 1
            line_out[line_out < 0.1] = 0
            edge_out[edge_out < 0.1] = 0

            for i in range(items['line_small'].shape[0]):
                line_ = (line_out[i, 0, ...].cpu().numpy() * 255).astype(np.uint8)
                edge_ = (edge_out[i, 0, ...].cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite(val_path_line_1024 + '/' + items['name'][i], line_)
                cv2.imwrite(val_path_edge_1024 + '/' + items['name'][i], edge_)
