import os
import torch
import torchvision.transforms as transforms
import torchvision.models as torchmodels
import numpy as np
import shutil

from utils import utils_detector
from dataset.dataloader import CustomDatasetFromImages
from constants import (
    base_dir_groundtruth,
    base_dir_detections_cd,
    base_dir_detections_fd,
    base_dir_metric_cd,
    base_dir_metric_fd,
    base_dir_counts,
    num_windows,
    num_actions
)


def save_args(__file__, args):
    shutil.copy(os.path.basename(__file__), args.cv_dir)
    with open(args.cv_dir + "/args.txt", "w") as f:
        f.write(str(args))


def get_detected_boxes(policy, file_dirs, metrics, set_labels, num_wind=num_windows):
    for index, file_dir_st in enumerate(file_dirs):
        counter = 0
        for xind in range(num_wind):
            for yind in range(num_wind):
                # ---------------- Read Ground Truth ----------------------------------
                outputs_all = []
                gt_path = "{}/{}_{}_{}.txt".format(
                    base_dir_groundtruth, file_dir_st, xind, yind
                )
                if os.path.exists(gt_path) and os.path.getsize(gt_path) > 0:
                    gt = np.loadtxt(gt_path).reshape([-1, 5])

                    targets = np.hstack((np.zeros((gt.shape[0], 1)), gt))

                    # ----------------- Read Detections -------------------------------
                    if policy[index, counter] == 1:
                        preds_dir = "{}/{}_{}_{}.npy".format(
                            base_dir_detections_fd, file_dir_st, xind, yind
                        )
                        if os.path.exists(preds_dir):
                            preds = np.load(preds_dir).reshape([-1, 7])
                            outputs_all.append(torch.from_numpy(preds))
                    else:
                        preds_dir = "{}/{}_{}_{}.npy".format(
                            base_dir_detections_cd, file_dir_st, xind, yind
                        )
                        if os.path.exists(preds_dir):
                            preds = np.load(preds_dir).reshape([-1, 7])
                            outputs_all.append(torch.from_numpy(preds))
                    set_labels += targets[:, 1].tolist()
                    metrics += utils_detector.get_batch_statistics(
                        outputs_all, torch.from_numpy(targets), 0.5
                    )
                else:
                    continue
                counter += 1

    return metrics, set_labels


def read_offsets(image_ids, num_actions):
    offset_fd = torch.zeros((len(image_ids), num_actions))
    offset_cd = torch.zeros((len(image_ids), num_actions))
    for index, img_id in enumerate(image_ids):
        offset_fd[index, :] = torch.from_numpy(
            np.load(f"{base_dir_metric_fd}/{img_id}.npy").flatten()
        )
        offset_cd[index, :] = torch.from_numpy(
            np.load(f"{base_dir_metric_cd}/{img_id}.npy").flatten()
        )

    return offset_fd, offset_cd


def read_counts(image_ids, num_actions):
    counts = torch.zeros((len(image_ids), num_actions))
    for index, img_id in enumerate(image_ids):
        counts[index, :] = torch.from_numpy(
            np.load(f"{base_dir_counts}/{img_id}.npy").flatten()
        )
    return counts


def performance_stats(policies, rewards):
    # Print the performace metrics including the average reward, average number
    # and variance of sampled num_patches, and number of unique policies
    policies = torch.cat(policies, 0)
    rewards = torch.cat(rewards, 0)

    reward = rewards.mean()
    num_unique_policy = policies.sum(1).mean()
    variance = policies.sum(1).std()

    policy_set = [p.cpu().numpy().astype(np.int).astype(np.str) for p in policies]
    policy_set = set(["".join(p) for p in policy_set])

    return reward, num_unique_policy, variance, policy_set


def compute_reward(offset_fd, offset_cd, object_counts, policy, beta, sigma):
    """
    Args:
        offset_fd: np.array, shape [batch_size, num_actions]
        offset_cd: np.array, shape [batch_size, num_actions]
        policy: np.array, shape [batch_size, num_actions], binary-valued (0 or 1)
        beta: scalar
        sigma: scalar
    """
    # Reward function favors policies that drops patches only if the classifier
    # successfully categorizes the image
    offset_cd += beta
    reward_patch_diff = (
        (offset_fd - offset_cd) * policy + -1 * ((offset_fd - offset_cd) * (1 - policy))
    ) * object_counts
    reward_patch_acqcost = (policy.size(1) - policy.sum(dim=1)) / policy.size(1)
    reward_img = reward_patch_diff.sum(dim=1) + sigma * reward_patch_acqcost
    reward = reward_img.unsqueeze(1)
    return reward.float()


def get_transforms(img_size):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform_train = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.RandomCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    return transform_train, transform_test


def get_dataset(img_size, root="data/", num_act=num_actions):
    transform_train, transform_test = get_transforms(img_size)
    trainset = CustomDatasetFromImages(root + "train.csv", transform_train, num_act=num_act)
    testset = CustomDatasetFromImages(root + "val.csv", transform_test, num_act=num_act)

    return trainset, testset


def set_parameter_requires_grad(model, feature_extracting):
    # When loading the models, make sure to call this function to update the weights
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def get_model(num_output):
    agent = torchmodels.resnet34(pretrained=True)
    set_parameter_requires_grad(agent, False)
    num_ftrs = agent.fc.in_features
    agent.fc = torch.nn.Linear(num_ftrs, num_output)

    return agent
