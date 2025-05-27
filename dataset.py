import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd # Import pandas as it's used in the class definition

class RepoSplitTimeSeriesDataset(Dataset):
    def __init__(self, X_df, y_series, meta_df, timesteps=5, mode="train", split_ratio=0.8):
        self.samples = []
        assert mode in ["train", "val"], "mode must be 'train' or 'val'"

        grouped = meta_df.groupby("Repo")

        for repo_id, group_indices in grouped.groups.items():
            # sort by time within each repo
            group = meta_df.loc[group_indices].sort_values("Scan date")
            sorted_idx = group.index.tolist()

            total = len(sorted_idx)
            split_point = int(total * split_ratio)

            if total < timesteps:
                continue  # skip short repos

            if mode == "train":
                use_idx = sorted_idx[:split_point]
            else:  # val
                use_idx = sorted_idx[split_point:]

            # re-slide within the selected portion
            for i in range(len(use_idx) - timesteps + 1):
                window_idx = use_idx[i:i + timesteps]
                x_seq = torch.tensor(X_df.loc[window_idx].values, dtype=torch.float32)
                y_seq = torch.tensor(y_series.loc[window_idx].values, dtype=torch.long)
                # meta_seq = meta_df.loc[window_idx].iloc[-1].to_dict() # This line was commented out in the notebook, keeping it commented
                self.samples.append((x_seq, y_seq))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
