from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from utils import save_best_record
from model import Model
from dataset import Dataset
from train import train
from test_10crop import test
import option
from tqdm import tqdm
from utils import Visualizer
from config import *
import os

viz = Visualizer(env='wrongway-rtfm', use_incoming_socket=False)

if __name__ == '__main__':
    args = option.parser.parse_args()
    config = Config(args)

    # Load Datasets
    train_nloader = DataLoader(Dataset(args, test_mode=False, is_normal=True),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)
    train_aloader = DataLoader(Dataset(args, test_mode=False, is_normal=False),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)

    test_loader = DataLoader(Dataset(args, test_mode=True),
                             batch_size=1, shuffle=False,
                             num_workers=0, pin_memory=True)

    # Model Setup
    model = Model(args.feature_size, args.batch_size)

    for name, value in model.named_parameters():
        print(name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')

    optimizer = optim.Adam(model.parameters(),
                           lr=config.lr[0], weight_decay=0.005)

    test_info = {"epoch": [], "test_AUC": []}
    best_AUC = -1
    output_path = ''  # Optional: Update if you want to save best-AUC logs

    print("\n✅ Starting Training...\n")

    for step in tqdm(
            range(1, args.max_epoch + 1),
            total=args.max_epoch,
            dynamic_ncols=True
    ):
        # ✅ Reset data iterators for each epoch
        loadern_iter = iter(train_nloader)
        loadera_iter = iter(train_aloader)

        # ✅ LR Scheduling
        if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr[step - 1]

        # ✅ Train one epoch
        train(loadern_iter, loadera_iter, model, args.batch_size, optimizer, viz, device)

        # ✅ Test after each epoch
        auc = test(test_loader, model, args, viz, device)
        test_info["epoch"].append(step)
        test_info["test_AUC"].append(auc)

        # ✅ Save best model
        if test_info["test_AUC"][-1] > best_AUC:
            best_AUC = test_info["test_AUC"][-1]
            torch.save(model.state_dict(), './ckpt/' + args.model_name + f'{step}-i3d.pkl')
            save_best_record(test_info, os.path.join(output_path, 'AUC-per-epoch', f'{step}-step-AUC.txt'))

    # ✅ Save final model after all epochs
    torch.save(model.state_dict(), './ckpt/' + args.model_name + 'final.pkl')
    print("\n✅ Training Complete. Final model saved.\n")
