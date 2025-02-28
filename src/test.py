#%%
import torch
import os
from torch.utils.data import DataLoader, Dataset

# Dummy dataset
class DummyDataset(Dataset):
    def __init__(self, size=1000):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        print(f"🔹 Process {os.getpid()} fetching index {idx}", flush=True)

        # 🔹 Catch any potential crashes
        try:
            return idx
        except Exception as e:
            print(f"🚨 Worker Error: {e}")
            return -1 
        
# Windows multiprocessing fix
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    print(f"Main process ID: {os.getpid()}")

    dataset = DummyDataset()
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=2)

    for batch in dataloader:
        print(f"✅ Batch: {batch}")
        break  # Test only the first batch


# %%
