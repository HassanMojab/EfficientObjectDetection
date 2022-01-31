"""
This function runs object detection on the given high resolution image.
How to Run on the xView Dataset:
    python detect.py \
        --img_path data/your_dataset/images/100.tif \
        --load_fpn model/fd_fpn \
        --load_cpn model/fd_cpn
"""
import torch
import numpy as np
from PIL import Image
import time

from utils import utils

import torch.backends.cudnn as cudnn

cudnn.benchmark = True

import argparse

parser = argparse.ArgumentParser(
    description="Detect objects using cascaded policy networks"
)
parser.add_argument("--img_path", help="Path of the image")
parser.add_argument("--img_size_cpn", type=int, default=448, help="CPN Image Size")
parser.add_argument("--img_size_fpn", type=int, default=112, help="FPN Image Size")
parser.add_argument("--load_cpn", help="checkpoint to load CPNet agent from")
parser.add_argument("--load_fpn", help="checkpoint to load FPNet agent from")
parser.add_argument(
    "--num_windows_cpn",
    type=int,
    default=4,
    help="Number of windows in one dimension for CPN",
)
parser.add_argument(
    "--num_windows_fpn",
    type=int,
    default=2,
    help="Number of windows in one dimension for FPN",
)
parser.add_argument(
    "--cv_dir",
    default="cv/tmp/",
    help="checkpoint directory (models and logs are saved here)",
)
parser.add_argument(
    "--vis_num", action="store_true", help="visualize with numbers",
)

args = parser.parse_args()

cuda_or_cpu = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(cuda_or_cpu)


def detect_cpn(inputs_cpn):
    with torch.no_grad():
        # Get the low resolution agent images
        probs = torch.sigmoid(agent_cpn(inputs_cpn))

        # Sample the policy from the agents output
        policy_cpn = probs.data.clone()
        policy_cpn[policy_cpn < 0.5] = 0.0
        policy_cpn[policy_cpn >= 0.5] = 1.0

        return policy_cpn


def detect_fpn(inputs_fpn):
    probs = torch.sigmoid(agent_fpn(inputs_fpn))
    # Sample the policy from the agents output
    policy_fpn = probs.data.clone()
    policy_fpn[policy_fpn < 0.5] = 0.0
    policy_fpn[policy_fpn >= 0.5] = 1.0

    return policy_fpn[0]


def detect():
    if args.load_cpn:
        agent_cpn.eval()
    if args.load_fpn:
        agent_fpn.eval()

    total_time = 0

    policy_cpn = torch.ones((1, args.num_windows_cpn ** 2))

    start = time.time()

    # Open image
    img_cpn = Image.open(args.img_path)
    # Transform the image
    _, transform_cpn = utils.get_transforms(args.img_size_cpn)
    inputs_cpn = transform_cpn(img_cpn)
    inputs_cpn = torch.unsqueeze(inputs_cpn, 0)

    inputs_cpn.to(device)

    if args.load_cpn:
        policy_cpn = detect_cpn(inputs_cpn)

    total_time += time.time() - start

    policy_fpn_list = []

    # Select the images to run Fine Detector on
    for xind in range(args.num_windows_cpn):
        for yind in range(args.num_windows_cpn):
            # Get the low resolution agent image
            # -----------------------------------------------
            policy_fpn = torch.zeros((args.num_windows_fpn ** 2))
            index_ft = xind * args.num_windows_cpn + yind
            if policy_cpn[:, index_ft] != 0:
                start = time.time()
                policy_fpn = torch.ones((args.num_windows_fpn ** 2))

                if args.load_fpn:
                    inputs_fpn = inputs_cpn[
                        :,
                        :,
                        xind * args.img_size_fpn : xind * args.img_size_fpn
                        + args.img_size_fpn,
                        yind * args.img_size_fpn : yind * args.img_size_fpn
                        + args.img_size_fpn,
                    ]
                    policy_fpn = detect_fpn(inputs_fpn)

                total_time += time.time() - start

            policy_fpn_list.append(policy_fpn.numpy())

    visualize_actions(policy_cpn.numpy(), policy_fpn_list)

    print(f"Run Time: {total_time:.3f}s")


PLOT_WIDTH = 32
FULL_BLOCK = "█"
HALF_BLOCK = "\u2593"
EMPTY_BLOCK = "\u2591"

if args.vis_num:
    FULL_BLOCK = "2 "
    HALF_BLOCK = "1 "
    EMPTY_BLOCK = "0 "


def print_row(row_actions, count):
    for _ in range(count if args.vis_num else count // 2):
        for i in range(args.num_windows_cpn * args.num_windows_fpn):
            if row_actions[i, 0]:
                if row_actions[i, 1]:
                    print(FULL_BLOCK * count, end="")
                else:
                    print(HALF_BLOCK * count, end="")
            else:
                print(EMPTY_BLOCK * count, end="")
        print()


def visualize_actions(policy_cpn, policy_fpn_list):
    num_windows = args.num_windows_cpn * args.num_windows_fpn
    fpn_matrix = [
        x.reshape((args.num_windows_fpn, args.num_windows_fpn)) for x in policy_fpn_list
    ]
    fpn_matrix = np.concatenate(fpn_matrix, axis=1)
    fpn_matrix = np.split(fpn_matrix, fpn_matrix.shape[1] // num_windows, axis=1)
    fpn_matrix = np.concatenate(fpn_matrix)
    cpn_matrix = policy_cpn.reshape(args.num_windows_cpn, args.num_windows_cpn)
    cpn_matrix = np.repeat(cpn_matrix, 2, axis=0)
    cpn_matrix = np.repeat(cpn_matrix, 2, axis=1)
    matrix = np.array([cpn_matrix, fpn_matrix]).transpose(1, 2, 0)

    count = 1 if args.vis_num else PLOT_WIDTH // num_windows

    for row in matrix:
        print_row(row, count)

    print(f"HR: {fpn_matrix.sum() / num_windows ** 2 * 100:.2f}%")


# --------------------------------------------------------------------------------------------------------#
if args.load_cpn:
    print(args.load_cpn)
    agent_cpn = utils.get_model(args.num_windows_cpn ** 2)
    checkpoint_cpn = torch.load(args.load_cpn, map_location=cuda_or_cpu)
    agent_cpn.load_state_dict(checkpoint_cpn["agent"])
    agent_cpn.to(device)

if args.load_fpn:
    agent_fpn = utils.get_model(args.num_windows_fpn ** 2)
    checkpoint_fpn = torch.load(args.load_fpn, map_location=cuda_or_cpu)
    agent_fpn.load_state_dict(checkpoint_fpn["agent"])
    agent_fpn.to(device)

detect()
