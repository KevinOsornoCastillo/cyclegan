import argparse
from data_generator import DataLoader
from tqdm import tqdm

""" Parametros por consola para organizar un experimento """
parser = argparse.ArgumentParser()
parser.add_argument('--norm', type=str, default='z_score', help='Normalizacion: min_max, z_score')
parser.add_argument('--num_patches', type=int, default=100, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='no root', help='root directory of the dataset')
parser.add_argument('--patch_size', type=int, default=32, help='size of patch')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
opt = parser.parse_args()


""" Obteniendo parametros del Data Generator y enviarlos a la clase DataLoader """
data_loader_params = {
    "norm": opt.norm,
    "num_patches": opt.num_patches,
    "path_data": opt.dataroot,
    "patch_size": opt.patch_size
}

data_generator = DataLoader(data_loader_params)




