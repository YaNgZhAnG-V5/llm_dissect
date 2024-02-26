import torch
from torch import nn
from torchvision import datasets, transforms
from dissect import ForwardADExtractor


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


def main():
    # for model training on MNIST, initialize model and data loader
    train_kwargs = {"batch_size": 256}
    test_kwargs = {"batch_size": 1000}
    use_cuda = True
    device = torch.device("cuda:1" if use_cuda else "cpu")
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
    forward_ad_extractor = ForwardADExtractor(model)
    input_tensor = next(iter(data_loader))[0].to(device)
    print(input_tensor.shape)
    input_grads = forward_ad_extractor.forward_ad(input_tensor)
    print(len(input_grads))
    for name, input_grad in input_grads.items():
        print(name, input_grad.shape)


if __name__ == "__main__":
    main()
