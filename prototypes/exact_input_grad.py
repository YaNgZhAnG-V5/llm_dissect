import torch


def input_grad(model, input_tensor, hook_objs):
    actual_grads = []
    input_tensor.requires_grad = True
    _ = model(input_tensor)

    for hook in hook_objs:
        # TODO fix shape
        actual_grad = []
        for i in range(hook.output.shape[1]):
            v = torch.zeros_like(hook.output)
            v[:, i] = 1
            input_tensor.grad = None
            hook.output.backward(v, retain_graph=True)
            actual_grad.append(input_tensor.grad.detach())
        actual_grads.append(torch.stack(actual_grad))
    return actual_grads
