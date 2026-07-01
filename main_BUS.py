

import numpy as np

from dataloader.dataloader import Busdata
from Model.Model import PINN
import argparse
import os
import torch


def set_seed(seed=42):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def load_data(args):
    root = r'E:\SOE\data_processing\Segment extraction results\bus_505Ah\Model_Features'
    data_engine = Busdata(root=root, args=args)

    all_files = sorted([os.path.join(root, f) for f in os.listdir(root) if f.startswith('Final_')])

    import random
    random.seed(42)
    random.shuffle(all_files)

    n_files = len(all_files)
    train_end = int(n_files * 0.8)
    val_end = int(n_files * 0.9)

    train_files = all_files[:train_end]
    val_files = all_files[train_end:val_end]
    test_files = all_files[val_end:]

   
    all_train_feats = []
    for f in train_files:
        feat, _ = data_engine.load_bus_battery(f)
        all_train_feats.append(feat)

    combined_train = np.vstack(all_train_feats)
    data_engine.feature_means = combined_train.mean(axis=0)
    data_engine.feature_stds = combined_train.std(axis=0)
    data_engine.feature_stds[data_engine.feature_stds == 0] = 1.0
    data_engine.feature_stds += 1e-8

    train_dict = data_engine.read_all(specific_path_list=train_files, is_test=False)
    val_dict = data_engine.read_all(specific_path_list=val_files, is_test=True)
    test_dict = data_engine.read_all(specific_path_list=test_files, is_test=True)

    loaders = {
        'train': train_dict['test_3'],
        'valid': val_dict['test_3'],
        'test': test_dict['test_3']
    }

    return loaders, data_engine, test_files


def main():
    set_seed(42)
    args = get_args()

    for e in range(1): 
        save_folder = f'results_of_reviewer/Bus_results/Experiment{e + 1}'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        log_dir = 'logging.txt'
        setattr(args, "save_folder", save_folder)
        setattr(args, "log_dir", log_dir)

        dataloader, data_engine, test_files = load_data(args)
        feature_means = data_engine.feature_means
        feature_stds = data_engine.feature_stds

        pinn = PINN(args, feature_means=feature_means, feature_stds=feature_stds)

        pinn.Train(trainloader=dataloader['train'],
                   validloader=dataloader['valid'],
                   testloader=dataloader['test'])

      

        pinn.solution_u.load_state_dict(pinn.best_model['solution_u'])
        pinn.eval()

        trace_folder = os.path.join(save_folder, 'test_traces')
        if not os.path.exists(trace_folder):
            os.makedirs(trace_folder)

        with torch.no_grad():
            for file_path in test_files:
                file_name = os.path.basename(file_path).replace('.csv', '')

                single_dict = data_engine.read_all(specific_path_list=[file_path], is_test=True)
                single_loader = single_dict['test_3']

                f_true, f_pred = [], []

                for x, _, y, _ in single_loader:
                    u = pinn.predict(x.to('cuda'))
                    f_true.append(y.numpy())
                    f_pred.append(u.cpu().numpy())

                res = np.hstack([np.concatenate(f_true), np.concatenate(f_pred)])
                np.save(os.path.join(trace_folder, f'{file_name}_trace.npy'), res)

        


def get_args():
    parser = argparse.ArgumentParser('Hyper Parameters for Bus dataset with PatchFormer-PINN')

    parser.add_argument('--data', type=str, default='Bus', help='XJTU, HUST, MIT, TJU, Bus')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--normalization_method', type=str, default='z-score')

    parser.add_argument('--epochs', type=int, default=300, help='epoch')
    parser.add_argument('--early_stop', type=int, default=100, help='early stop')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_F', type=float, default=1e-4)

    parser.add_argument('--alpha', type=float, default=0.3, help='PDE loss weight ')
    parser.add_argument('--beta', type=float, default=0.1, help='physical constraint weight')

    parser.add_argument('--seq_len', type=int, default=8)
    parser.add_argument('--patch_len', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--e_layers', type=int, default=1)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--F_layers_num', type=int, default=3)
    parser.add_argument('--F_hidden_dim', type=int, default=64)
    parser.add_argument('--log_dir', type=str, default='text_log.txt')
    parser.add_argument('--save_folder', type=str, default='results/Bus_results')
    parser.add_argument('--dropout', type=float, default=0.5)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
