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

    def __getitem__(self, index: int) -> Tuple[Union[Tensor, Dict[str, Tensor]], Union[Tensor, Dict[str, Tensor]]]:
        in_out_dict = torch.load(osp.join(self.root, self.files[index]), map_location="cpu")
        inputs = in_out_dict["input"]
        # When saving the layer inputs outputs, the tensors contain batch dimension.
        # Now we need to get rid of the batch dimension.
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.squeeze(0)
        else:
            for k, v in inputs.items():
                inputs.update({k: v.squeeze(0)})

        outputs = in_out_dict["output"]
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.squeeze(0)
        else:
            for k, v in outputs.items():
                outputs.update({k: v.squeeze(0)})

        return inputs, outputs
