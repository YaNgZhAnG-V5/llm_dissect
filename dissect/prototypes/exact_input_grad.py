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
            actual_grad.append(input_tensor.grad.sum().detach())
        actual_grads.append(torch.stack(actual_grad))
    return actual_grads


def input_grad_high_dim(model, input_tensor, hook_objs):
    actual_grads = []
    input_tensor.requires_grad = True
    _ = model(input_tensor)

    for hook in hook_objs:
        # TODO fix shape
        actual_grad = []
        grad_shape = (hook.output.shape + input_tensor.shape)[1:]
        dims = [torch.arange(size) for size in hook.output[0, :].shape]
        all_indices = torch.cartesian_prod(*dims)
        for index in all_indices:
            v = torch.zeros_like(hook.output)
            # TODO now always assume batch size = 1
            v[(0, ) + tuple(index.tolist())] = 1
            input_tensor.grad = None
            hook.output.backward(v, retain_graph=True)
            actual_grad.append(input_tensor.grad.mean().detach())
        actual_grads.append(torch.stack(actual_grad).reshape(grad_shape))
    return actual_grads
