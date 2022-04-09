from dataset import SketchDataset
from model import *
from shutil import copyfile
from utils import *
import argparse
import os
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='test', help='The name of this exp')
    parser.add_argument('--config_file', type=str, default='configs/config.yml')
    parser.add_argument('--GPU_ids', type=str, default='0')
    parser.add_argument('--ckpt_path', type=str, default='./ckpt')

    args = parser.parse_args()
    args.ckpt_path = os.path.join(args.ckpt_path, args.name)
    os.makedirs(args.ckpt_path, exist_ok=True)
    config_path = os.path.join(args.ckpt_path, 'config.yml')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU_ids
    copyfile(args.config_file, config_path)
    config = Config(config_path)

    set_seed(42)
    device = torch.device("cuda")
    train_dataset = SketchDataset(config, config.train_flist, augment=True, training=True)
    val_dataset = SketchDataset(config, config.val_flist, augment=False, training=False)

    train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=True,
                              batch_size=config.batch_size, num_workers=16)
    val_loader = DataLoader(val_dataset, shuffle=False, pin_memory=True,
                            batch_size=config.batch_size, num_workers=4)
    sample_iterator = val_dataset.create_iterator(config.sample_size)

    # model = StructureUpsampling()
    model = StructureUpsampling4()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    val_path = os.path.join(args.ckpt_path, 'validation')
    samples_path = os.path.join(args.ckpt_path, 'samples')
    create_dir(val_path)
    create_dir(samples_path)

    epoch = 0
    keep_training = True
    max_iteration = int(float((config.max_iters)))
    total = len(train_dataset)
    iteration = 0
    best_loss = 9999

    while keep_training:
        epoch += 1
        progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])

        for items in train_loader:
            model.train()
            iteration += 1

            for k in items:
                if k != 'name':
                    items[k] = items[k].to(device)

            line_out, line_in = model.forward(items['line_small'])

            line_loss1 = F.binary_cross_entropy_with_logits(line_out, items['line_large'])
            line_loss2 = F.binary_cross_entropy_with_logits(line_in, items['line_small_gt'])

            optimizer.zero_grad()
            loss = line_loss1 + line_loss2
            loss.backward()
            optimizer.step()
            progbar.add(len(items['edge_small']), values=[("epoch", epoch),
                                                          ("iter", iteration),
                                                          ("loss", loss.item())])

            if iteration % config.sample_interval == 0:
                model.eval()
                with torch.no_grad():
                    items = next(sample_iterator)
                    for k in items:
                        if k != 'name':
                            items[k] = items[k].to(device)

                    edge_out, edge_in = model.forward(items['edge_small'])
                    line_out, line_in = model.forward(items['line_small'])
                    edge_out = torch.sigmoid(edge_out)
                    line_out = torch.sigmoid(line_out)
                    edge_in = torch.sigmoid(edge_in)
                    line_in = torch.sigmoid(line_in)

                image_per_row = 2 if config.sample_size > 6 else 1
                images = stitch_images(postprocess(items['edge_small'].cpu(), size=512),
                                       postprocess(items['edge_large'].cpu(), size=512),
                                       postprocess(edge_out.cpu(), size=512),
                                       postprocess(edge_in.cpu(), size=512),
                                       postprocess(items['line_small'].cpu(), size=512),
                                       postprocess(items['line_large'].cpu(), size=512),
                                       postprocess(line_out.cpu(), size=512),
                                       postprocess(line_in.cpu(), size=512),
                                       img_per_row=image_per_row)
                name = os.path.join(samples_path, str(iteration).zfill(5) + ".jpg")
                print('\nsaving sample ' + name)
                images.save(name)

            if iteration % config.save_interval == 0:
                torch.save({'iteration': iteration, 'model': model.state_dict(),
                            'optimizer': optimizer.state_dict()}, os.path.join(args.ckpt_path, 'last.pth'))

            if iteration % config.eval_interval == 0:
                model.eval()
                eval_losses = []
                with torch.no_grad():
                    for items in tqdm(val_loader):
                        for k in items:
                            if k != 'name':
                                items[k] = items[k].to(device)

                        line_out, line_in = model.forward(items['line_small'])
                        line_loss1 = F.binary_cross_entropy_with_logits(line_out, items['line_large'])
                        line_loss2 = F.binary_cross_entropy_with_logits(line_in, items['line_small_gt'])
                        loss = line_loss1 + line_loss2
                        eval_losses.append(loss.item())

                eval_loss = np.mean(eval_losses)
                if eval_loss < best_loss:
                    best_loss = eval_loss
                    torch.save({'iteration': iteration, 'model': model.state_dict(), 'eval_loss': best_loss,
                                'optimizer': optimizer.state_dict()}, os.path.join(args.ckpt_path, 'best.pth'))
                print('Eval loss:', eval_loss, 'Best loss:', best_loss)

            if iteration > max_iteration:
                break
