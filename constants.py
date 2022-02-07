base_dir_detections_fd = "./data/your_dataset/base_dir_detections_fd"  # Directory that contains detections by fine level detector
base_dir_detections_cd = "./data/your_dataset/base_dir_detections_cd"  # Directory that contains detections by coarse level detector
base_dir_groundtruth = "./data/your_dataset/base_dir_groundtruth"  # Directory that contains ground truth bounding boxes
base_dir_metric_fd = "./data/your_dataset/base_dir_metric_fd"  # Directory that contains AP or AR values by the fine detector
base_dir_metric_cd = "./data/your_dataset/base_dir_metric_cd"  # Directory that contains AP or AR values by the coarse detector
base_dir_counts = "./data/your_dataset/base_dir_counts"  # Directory that contains number of objects in the patch
num_actions = 16  # Hyperparameter, should be equal to num_windows * num_windows
num_windows = 4  # Number of windows in one dimension
