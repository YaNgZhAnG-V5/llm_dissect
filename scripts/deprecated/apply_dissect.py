import os.path as osp
from argparse import ArgumentParser

import mmengine
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from mmengine.runner import set_random_seed
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from dissect.dissectors import Dissector
from dissect.utils import Device


def parse_args():
    parser = ArgumentParser("Apply dissection.")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID.")
    parser.add_argument("--work-dir", "-w", default="workdirs/debug/", help="Working directory to store output files.")

    return parser.parse_args()


class MLP(nn.Module):

    def __init__(self, num_units_hidden=1024, num_classes=10):
        super().__init__()
        self.linear_1 = nn.Linear(784, num_units_hidden)
        self.relu_1 = nn.ReLU()
        self.linear_2 = nn.Linear(num_units_hidden, num_units_hidden)
        self.relu_2 = nn.ReLU()
        self.linear_3 = nn.Linear(num_units_hidden, num_classes)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        out = self.relu_1(self.linear_1(x))
        logits = self.linear_3(self.relu_2(self.linear_2(out)))
        return logits


def train(model: nn.Module, device: Device) -> None:
    train_loader_kwargs = {"batch_size": 256}

    data_loader_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
    train_loader_kwargs.update(data_loader_kwargs)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    data_loader = DataLoader(dataset, **train_loader_kwargs)
    # train model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(3):
        model.train()
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(dataset)} "
                    f"({100. * batch_idx / len(dataset):.0f}%)]\tLoss: {loss.item():.6f}"
                )


def main():
    set_random_seed(42)

    args = parse_args()
    work_dir = args.work_dir
    mmengine.mkdir_or_exist(work_dir)
    # for model training on MNIST, initialize model and data loader
    device = torch.device(f"cuda:{args.gpu_id}")
    data_loader_cfg = {"batch_size": 256, "num_workers": 1, "pin_memory": True, "shuffle": True}

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    data_loader = DataLoader(dataset, **data_loader_cfg)
    model = MLP()
    model.to(device)

    # get dissect result from the randomly initialized model
    dissector = Dissector(model)
    data_points = next(iter(data_loader))
    input_tensor, target = data_points[0].to(device), data_points[1].to(device)
    criterion = nn.CrossEntropyLoss()
    dissect_ret = dissector.dissect(input_tensor=input_tensor, target=target, criterion=criterion)

    # get dissect result from the trained model
    train(model, device)
    dissect_ret_trained = dissector.dissect(input_tensor=input_tensor, target=target, criterion=criterion)
    for dissect_item_name, dissect_item in dissect_ret.items():
        # skip biases for simplicity
        if dissect_item_name == "biases":
            continue

        for layer_name, result in dissect_item.items():
            plt.figure()
            if dissect_item_name == "weights":
                result_trained = dissect_ret_trained[dissect_item_name]["weights"][layer_name]

                # sum over input dim
                result_trained = result_trained.mean(-1).numpy()
                result = result.mean(-1).numpy()
            else:
                result_trained = dissect_ret_trained[dissect_item_name][layer_name]
                result_trained = result_trained.mean(0).numpy()
                result = result.mean(0).numpy()
            plt.plot(result_trained, label="trained")
            plt.plot(result, label="untrained")
            plt.legend()
            plt.title(f"{dissect_item_name}_{layer_name}")
            plt.savefig(osp.join(work_dir, "{dissect_item_name}_{layer_name}.png"))

    print(f"Output files have been saved to: {work_dir}")


if __name__ == "__main__":
    main()
