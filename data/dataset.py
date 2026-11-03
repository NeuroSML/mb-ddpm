import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MultiBridgeDataset(Dataset):
    """
    Dataset for on-the-fly loading:
        data_dir/mode/source_name/*.npy  -> source
        data_dir/mode/target_name/*.npy  -> target
    Each source file must have a matching target file by name.
    """

    def __init__(
        self,
        data_dir,
        source_name,
        target_name,
    ):

        target_dir = os.path.join(data_dir, target_name)

        # List all target files
        target_files = sorted(os.listdir(target_dir))

        # Keep only files that exist in target folder
        self.samples = []
        for fname in target_files:
            tgt_path = os.path.join(target_dir, fname)
            extension = os.path.splitext(fname)[1]

            # Extract prefix to find corresponding source files
            prefix = fname.rsplit("_", 1)[0]

            # Find corresponding source files for all source modalities
            src_paths = []
            for name in source_name:
                source_dir = os.path.join(data_dir, name)
                src_path = os.path.join(source_dir, f"{prefix}_{name}{extension}")

                # Check if source file exists
                if os.path.exists(src_path):
                    src_paths.append(src_path)
                else:
                    raise FileNotFoundError(
                        f"Source file {src_path} not found for target file {tgt_path}."
                    )

            self.samples.append((tuple(src_paths), tgt_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src_paths, tgt_path = self.samples[idx]

        xT = []
        for src_path in src_paths:
            xT.append(self.load(src_path))  # source
        xT = torch.cat(xT, dim=0)
        x0 = self.load(tgt_path)  # target

        sample_id = os.path.basename(tgt_path).rsplit("_", 1)[0]

        return x0, xT, sample_id

    @staticmethod
    def load(path):
        if path.endswith(".npy"):
            arr = np.load(path)
            arr = torch.from_numpy(arr).float()
        elif path.endswith("JPEG") or path.endswith("jpg") or path.endswith("png"):
            img = Image.open(path).convert("RGB")
            arr = transforms.ToTensor()(img)
        else:
            raise ValueError(f"Unsupported file extension for path: {path}")
        if arr.ndim == 2:
            arr = arr[None, :, :]  # add channel dim -> (1,H,W)
        return arr
