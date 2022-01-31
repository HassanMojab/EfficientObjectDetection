"""
This function pretrains the policy network using the high resolution classifier
output-explained as pretraining the policy network in the paper.
How to Run on the xView Dataset:
    python test.py \
        --data_dir data/your_dataset/ \
        --load_fpn model/fpn \
        --load_cpn model/cpn
"""
import torch
import torch.utils.data as torchdata
import numpy as np
import tqdm
import time

from utils import utils, utils_detector

import torch.backends.cudnn as cudnn

cudnn.benchmark = True

import argparse

parser = argparse.ArgumentParser(description="Test Cascaded Policy Networks")
parser.add_argument("--data_dir", default="data/", help="data directory")
parser.add_argument("--img_size_cpn", type=int, default=448, help="CPN Image Size")
parser.add_argument("--img_size_fpn", type=int, default=112, help="FPN Image Size")
parser.add_argument(
    "--load_cpn", default=None, help="checkpoint to load CPNet agent from"
)
parser.add_argument(
    "--load_fpn", default=None, help="checkpoint to load FPNet agent from"
)
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
parser.add_argument("--batch_size", type=int, default=256, help="batch size")
parser.add_argument("--num_workers", type=int, default=2, help="Number of Workers")
args = parser.parse_args()

cuda_or_cpu = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(cuda_or_cpu)


def test():
    # Test the policy network
    if args.load_cpn:
        agent_cpn.eval()
    if args.load_fpn:
        agent_fpn.eval()

    metrics, set_labels, num_total, num_sampled = [], [], 0.0, 0.0

    total_time = 0

    with torch.no_grad():
        for (inputs_cpn, targets, _, _, _) in tqdm.tqdm(
            testloader, total=len(testloader)
        ):
            start = time.time()

            inputs_cpn = inputs_cpn.to(device)
            policy_cpn = torch.ones(
                (inputs_cpn.shape[0], args.num_windows_cpn ** 2), device=device
            )

            if args.load_cpn:
                # Get the low resolution agent images
                probs = torch.sigmoid(agent_cpn(inputs_cpn))

                # Sample the policy from the agents output
                policy_cpn = probs.data.clone()
                policy_cpn[policy_cpn < 0.5] = 0.0
                policy_cpn[policy_cpn >= 0.5] = 1.0

            total_time += time.time() - start

            # Select the images to run Fine Detector on
            for xind in range(args.num_windows_cpn):
                for yind in range(args.num_windows_cpn):
                    # Get the low resolution agent image
                    # -----------------------------------------------
                    policy_fpn = torch.zeros(
                        (policy_cpn.shape[0], args.num_windows_fpn ** 2), device=device
                    )
                    index_ft = xind * args.num_windows_cpn + yind
                    selected_indices = policy_cpn[:, index_ft] != 0

                    if selected_indices.any():
                        start = time.time()
                        policy_fpn[selected_indices] = torch.ones(
                            (selected_indices.sum(), args.num_windows_fpn ** 2),
                            device=device,
                        )

                        if args.load_fpn:
                            inputs_fpn = inputs_cpn[
                                selected_indices,
                                :,
                                xind * args.img_size_fpn : xind * args.img_size_fpn
                                + args.img_size_fpn,
                                yind * args.img_size_fpn : yind * args.img_size_fpn
                                + args.img_size_fpn,
                            ]
                            probs = torch.sigmoid(agent_fpn(inputs_fpn))
                            # Sample the policy from the agents output
                            policy_fpn[selected_indices] = probs.data.clone()
                            policy_fpn[policy_fpn < 0.5] = 0.0
                            policy_fpn[policy_fpn >= 0.5] = 1.0

                        total_time += time.time() - start

                    num_sampled += (policy_fpn == 1).sum().cpu().numpy().tolist()
                    num_total += policy_fpn.size(1)

                    # Compute the Batch-wise metrics
                    targets_ind = [
                        "{}_{}_{}".format(
                            str(targets[0].numpy().tolist()), str(xind), str(yind)
                        )
                    ]

                    metrics, set_labels = utils.get_detected_boxes(
                        policy_fpn,
                        targets_ind,
                        metrics,
                        set_labels,
                        num_wind=args.num_windows_fpn,
                    )

    # Compute the Precision and Recall Performance of the Agent and Detectors
    true_positives, pred_scores, pred_labels = [
        np.concatenate(x, 0) for x in list(zip(*metrics))
    ]
    _, recall, AP, _, _ = utils_detector.ap_per_class(
        true_positives, pred_scores, pred_labels, set_labels
    )

    print(
        f"Test - AP: {AP[0]:.3f} | AR: {recall.mean():.3f} | RS: {num_sampled / num_total:.3f}% | Run-time: {total_time:.3f}s"
    )


# --------------------------------------------------------------------------------------------------------#
_, testset = utils.get_dataset(args.img_size_cpn, args.data_dir)
testloader = torchdata.DataLoader(
    testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
)

# ---- Load the pre-trained model ----------------------
if args.load_cpn:
    agent_cpn = utils.get_model(args.num_windows_cpn ** 2)
    checkpoint_cpn = torch.load(args.load_cpn, map_location=cuda_or_cpu)
    agent_cpn.load_state_dict(checkpoint_cpn["agent"])
    agent_cpn.to(device)

if args.load_fpn:
    agent_fpn = utils.get_model(args.num_windows_fpn ** 2)
    checkpoint_fpn = torch.load(args.load_fpn, map_location=cuda_or_cpu)
    agent_fpn.load_state_dict(checkpoint_fpn["agent"])
    agent_fpn.to(device)

test()
