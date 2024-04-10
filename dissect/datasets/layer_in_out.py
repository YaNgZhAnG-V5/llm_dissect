import os
import os.path as osp
from typing import Dict, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset


class LayerInOutDataset(Dataset):

    def __init__(self, root: str) -> None:
        super().__init__()

        self.root = root
        self.files = os.listdir(self.root)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[Union[Tensor, Dict[str, Tensor]], Tensor]:
        in_out_dict = torch.load(osp.join(self.root, self.files[index]), map_location="cpu")
        return in_out_dict["input"], in_out_dict["output"]
