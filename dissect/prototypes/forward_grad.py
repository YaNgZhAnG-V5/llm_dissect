import torch
from torch import nn
import torch.autograd.forward_ad as fwAD
from torchvision import datasets, transforms
from torchvision.models import resnet50
from tqdm import tqdm

from dissect.prototypes.exact_input_grad import input_grad


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


class DualHookRegister:
    def __init__(self):
        self.output = None

    def output2dual(self, module, input, output):
        self.output = output
        return output

    def __call__(self):
        return self.output2dual


def get_layers(model):
    """get layers with trainable parameters"""
    params_name = list(dict(model.named_parameters()).keys())
    params_name = {".".join(name.split(".")[:-1]) for name in params_name}
    layers = []
    for name in params_name:
        name = name.split(".")
        layer = model
        for submodule_name in name:
            if submodule_name in [str(i) for i in range(100)]:
                layer = layer[int(submodule_name)]
            elif hasattr(layer, submodule_name):
                layer = getattr(layer, submodule_name)
        layers.append(layer)
    return layers


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

    # # train model
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # criterion = nn.CrossEntropyLoss()
    # for epoch in range(3):
    #     model.train()
    #     for batch_idx, (data, target) in enumerate(data_loader):
    #         data, target = data.to(device), target.to(device)
    #         optimizer.zero_grad()
    #         output = model(data)
    #         loss = criterion(output, target)
    #         loss.backward()
    #         optimizer.step()
    #         if batch_idx % 100 == 0:
    #             print(
    #                 f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(dataset)} "
    #                 f"({100. * batch_idx / len(dataset):.0f}%)]\tLoss: {loss.item():.6f}"
    #             )

    # add hook on all layers
    layers = get_layers(model)
    hook_list = []
    hook_objs = []
    for layer in layers:
        register_dual_hook = DualHookRegister()
        hook_objs.append(register_dual_hook)
        hook_list.append(layer.register_forward_hook(register_dual_hook()))

    output_forward_grads = []
    flag = False
    batch_size = 1
    for i in tqdm(range(batch_size)):
        with fwAD.dual_level():
            # make input a dual tensor
            # TODO fix shape
            input_tensor = dataset[0][0].to(device)
            # input_tensor = torch.ones(1, 10).to(device)
            tangent = torch.ones_like(input_tensor)
            input_tensor = fwAD.make_dual(input_tensor, tangent)

            # perform forward and collect forward gradient
            _ = model(input_tensor)
            for idx, hook in enumerate(hook_objs):
                output_jvp = fwAD.unpack_dual(hook.output).tangent.detach().squeeze(0)
                output_forward_grad = []
                for jvp_val in output_jvp:
                    output_forward_grad.append(jvp_val)
                output_forward_grad = torch.stack(output_forward_grad, dim=0)
                if not flag:
                    output_forward_grads.append(output_forward_grad)
                else:
                    output_forward_grads[idx] += output_forward_grad
        flag = True
    print("fowrad grad: shape")
    for i, forward_grad in enumerate(output_forward_grads):
        print(forward_grad.shape)
        output_forward_grads[i] = forward_grad / batch_size

    # get actual gradient
    actual_grads = input_grad(model, input_tensor, hook_objs)
    print("actual grad: shape")
    for actual_grad in actual_grads:
        print(actual_grad.shape)

    # compare forward grad with actual grad
    for forward_grad, actual_grad in zip(output_forward_grads, actual_grads):
        diff = torch.norm(forward_grad - actual_grad).item()
        print(f"diff: {diff:5f}")
        print(f"relative diff: {diff / (torch.norm(actual_grad).item() + 1e-8):5f}")
        abs_diff = torch.norm(forward_grad.abs() - actual_grad.abs()).item()
        print(f"abs diff: {abs_diff:5f}")
        print(f"relative abs diff: {abs_diff / (torch.norm(actual_grad.abs()).item() + 1e-8):5f}")
        print("")

    # compare ranking of forward grad with actual grad
    for forward_grad, actual_grad in zip(output_forward_grads, actual_grads):
        forward_grad_rank = torch.argsort(forward_grad.flatten(), descending=True)
        actual_grad_rank = torch.argsort(actual_grad.flatten(), descending=True)
        asb_rank_diff = (forward_grad_rank - actual_grad_rank).abs().float().mean()
        print(f"abs grad rank diff: {asb_rank_diff:5f}")
        forward_grad_rank = torch.argsort(forward_grad.flatten().abs(), descending=True)
        actual_grad_rank = torch.argsort(actual_grad.flatten().abs(), descending=True)
        asb_rank_diff = (forward_grad_rank - actual_grad_rank).abs().float().mean()
        print(f"abs grad magnitude rank diff: {asb_rank_diff:5f}")

    # put grad into bins, compare grad mismatch
    num_bins = [2, 4, 10]
    forward_grad_rank = torch.argsort(forward_grad.flatten().abs(), descending=True)
    actual_grad_rank = torch.argsort(actual_grad.flatten().abs(), descending=True)
    for bins in num_bins:
        mismatch = 0
        for i in range(bins):
            forward_grad_bin = forward_grad_rank[
                (i * forward_grad_rank.shape[0] // bins) : ((i + 1) * forward_grad_rank.shape[0] // bins)
            ]
            actual_grad_bin = actual_grad_rank[
                (i * actual_grad_rank.shape[0] // bins) : ((i + 1) * actual_grad_rank.shape[0] // bins)
            ]
            for rank in forward_grad_bin:
                if rank not in actual_grad_bin:
                    mismatch += 1
        print(f"grad mismatch: {mismatch/forward_grad_rank.shape[0]:5f}")


if __name__ == "__main__":
    main()
