import torch
from torch.utils.data import Dataset
from typing import List, Dict

class ClinicalTrajectoryDataset(Dataset):
    def __init__(self, records: List[Dict]):
        self.data = records

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        r = self.data[i]
        return {
            "obs":      torch.from_numpy(r["obs"]).float(),
            "actions":  torch.from_numpy(r["actions"]).long(),
            "next_obs": torch.from_numpy(r["next_obs"]).float(),
            "outcomes": torch.from_numpy(r["outcomes"]).float(),
            "pathology_id": torch.tensor(r["pathology_id"], dtype=torch.long),
        }
