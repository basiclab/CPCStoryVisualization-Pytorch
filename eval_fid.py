import argparse
import os

from inference import Infer
from tensorboard import summary
from tensorboardX import SummaryWriter

from fid.vfid_score import fid_score as vfid_score
from fid.fid_score_v import fid_score

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate fid')
    parser.add_argument('--model_dir', dest='trained model')
    parser.add_argument('--vfid', type=str, default=True)
    parser.add_argument('--fid', type=str, default=True)
    args = parser.parse_args()





args = parse_args()

log_dir = os.path.join(args.model_dir, 'log')
logger = SummaryWriter(log_dir)

algo = Infer(output_dir, 1.0, args.load_ckpt)
algo.

if args.fid:




if args.vfid:


