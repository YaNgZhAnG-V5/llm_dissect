import torch
from torch import nn
from torchvision import datasets, transforms
from dissect.dissectors import ForwardADExtractor, Dissector
from matplotlib import pyplot as plt


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


def train(model, dataset, device):
    train_kwargs = {"batch_size": 256}
    test_kwargs = {"batch_size": 1000}
    use_cuda = True
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, **train_kwargs)
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


def input_grad(model, dataset, data_loader, device):
    # get input gradient
    forward_ad_extractor = ForwardADExtractor(model)
    input_tensor = next(iter(data_loader))[0].to(device)
    print(input_tensor.shape)
    input_grads = forward_ad_extractor.forward_ad(input_tensor)
    print(len(input_grads))

    # train model and get input gradient
    train(model, dataset, device)
    input_grads_trained = forward_ad_extractor.forward_ad(input_tensor)
    for name, input_grad in input_grads.items():
        print(name, input_grad.shape)
        plt.figure()
        input_grad_trained = input_grads_trained[name]
        plt.plot(input_grad_trained.mean(0).numpy(), label="trained")
        plt.plot(input_grad.mean(0).numpy(), label="untrained")
        plt.legend()
        plt.savefig(f"workdir/{name}_input_grad.png")


def main():
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for model training on MNIST, initialize model and data loader
    train_kwargs = {"batch_size": 256}
    test_kwargs = {"batch_size": 1000}
    use_cuda = True
    device = torch.device("cuda:2" if use_cuda else "cpu")
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, **train_kwargs)
    model = MLP()
    model.to(device)

    # get input gradient
    # input_grad(model, dataset, data_loader, device)

    # get dissect result
    dissector = Dissector(model)
    data_points = next(iter(data_loader))
    input_tensor, target = data_points[0].to(device), data_points[1].to(device)
    criterion = nn.CrossEntropyLoss()
    dissect_ret = dissector.dissect(input_tensor, target, criterion)

    # train model and get input gradient
    train(model, dataset, device)
    dissect_ret_trained = dissector.dissect(input_tensor, target, criterion)
    for dissect_item_name, dissect_item in dissect_ret.items():
        if dissect_item_name == "weights":
            dissect_item = dissect_item["weights"]
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
            plt.savefig(f"workdir/{dissect_item_name}_{layer_name}.png")


if __name__ == "__main__":
    main()
