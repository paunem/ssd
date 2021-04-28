import os
import shutil
from argparse import ArgumentParser

import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.model import SSD
from src.utils import generate_dboxes, Encoder
from src.transform import SimpleTransformer
from src.loss import Loss
from src.process import train, evaluate
from src.dataset import collate_fn, OIDataset


def get_args():
    parser = ArgumentParser(description="Implementation of SSD")
    parser.add_argument("--save-folder", type=str, default="trained_models",
                        help="path to folder containing model checkpoint file")
    parser.add_argument("--log-path", type=str, default="tensorboard/SSD")

    parser.add_argument("--epochs", type=int, default=50, help="number of total epochs to run")
    parser.add_argument("--batch-size", type=int, default=4, help="number of samples for each iteration")
    parser.add_argument("--multistep", nargs="*", type=int, default=[5, 7],
                        help="epochs at which to decay learning rate")

    parser.add_argument("--lr", type=float, default=2.6e-3, help="initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum argument for SGD optimizer")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="momentum argument for SGD optimizer")
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=4)

    args = parser.parse_args()
    return args


def main(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    train_params = {"batch_size": opt.batch_size,
                    "shuffle": True,
                    "drop_last": False,
                    "num_workers": opt.num_workers,
                    "collate_fn": collate_fn}

    eval_params = {"batch_size": opt.batch_size,
                   "shuffle": True,
                   "drop_last": False,
                   "num_workers": opt.num_workers,
                   "collate_fn": collate_fn}

    dboxes = generate_dboxes()
    model = SSD()
    train_set = OIDataset(SimpleTransformer(dboxes), train=True)
    train_loader = DataLoader(train_set, **train_params)
    val_set = OIDataset(SimpleTransformer(dboxes, eval=True), validation=True)
    val_loader = DataLoader(val_set, **eval_params)

    encoder = Encoder(dboxes)

    opt.lr = opt.lr * (opt.batch_size / 32)
    criterion = Loss(dboxes)

    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum,
                                weight_decay=opt.weight_decay,
                                nesterov=True)
    scheduler = MultiStepLR(optimizer=optimizer, milestones=opt.multistep, gamma=0.1)

    if torch.cuda.is_available():
        model.cuda()
        criterion.cuda()

    model = torch.nn.DataParallel(model)

    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)

    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    checkpoint_path = os.path.join(opt.save_folder, "SSD.pth")

    writer = SummaryWriter(opt.log_path)

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        first_epoch = checkpoint["epoch"] + 1
        model.module.load_state_dict(checkpoint["model_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        first_epoch = 0

    for epoch in range(first_epoch, opt.epochs):
        train(model, train_loader, epoch, writer, criterion, optimizer, scheduler)
        evaluate(model, val_loader, encoder, opt.nms_threshold)

        checkpoint = {"epoch": epoch,
                      "model_state_dict": model.module.state_dict(),
                      "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict()}
        torch.save(checkpoint, checkpoint_path)


if __name__ == "__main__":
    opt = get_args()
    main(opt)
