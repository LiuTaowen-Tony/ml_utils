import torch


def dict_abs(x):
    assert isinstance(x, dict)

    for key in x:
        x[key] = x[key].abs()


def dict_imul(x, a):
    assert isinstance(x, dict)
    assert isinstance(a, (int, float))

    for key in x:
        x[key] *= a


def dict_iadd(x, y):
    assert isinstance(x, dict)
    assert isinstance(y, dict)

    for key in y:
        if key not in x:
            x[key] = y[key].clone()
        else:
            x[key] += y[key]


def dict_iminus(x, y):
    assert isinstance(x, dict)
    assert isinstance(y, dict)

    for key in y:
        x[key] -= y[key]


def dict_isqr(x):
    assert isinstance(x, dict)

    for key in x:
        x[key] *= x[key]


def dict_isqrt(x):
    assert isinstance(x, dict)

    for key in x:
        x[key] = x[key] ** 0.5


def retrieve_param_grad_from_model(model):
    # assume model has grads
    grad_dict = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_dict[name] = param.grad.detach()
    return grad_dict


def compute_grad_dict_dataset(model, compute_grad_dict_fn, val_loader):
    bs = val_loader.batch_size
    num_samples = len(val_loader.dataset)
    # assert num_samples % bs == 0
    num_batches = num_samples // bs

    accumulated_grads = {}
    model.zero_grad()

    for i, input in enumerate(val_loader):
        if i >= num_batches:
            break
        grad_dict = compute_grad_dict_fn(model, input)
        dict_iadd(accumulated_grads, grad_dict)

    dict_imul(accumulated_grads, 1.0 / num_batches)
    model.zero_grad()
    return accumulated_grads


def compute_batch_grad_bias_std(
    model, compute_grad_dict_fn, val_loader, correct_grad, repeat=10
):
    bias_grad = {}
    std_grad = {}
    cnt = 0

    for input, target in val_loader:
        for i in range(repeat):
            grad_dict = compute_grad_dict_fn(model, input, target)
            dict_iadd(bias_grad, grad_dict)

            dict_iminus(grad_dict, correct_grad)
            dict_isqr(grad_dict)
            dict_iadd(std_grad, grad_dict)
            cnt += 1

    dict_imul(bias_grad, 1.0 / cnt)
    dict_iminus(bias_grad, correct_grad)

    dict_imul(std_grad, 1.0 / cnt)
    dict_isqrt(std_grad)

    return bias_grad, std_grad


def grad_dict_to_vector(grad_dict):
    l = []
    for value in grad_dict.values():
        if isinstance(value, torch.Tensor):
            l.append(value.view(-1))
        else:
            raise ValueError(f"Unsupported type: {type(value)}")
    if len(l) == 0:
        raise ValueError("No gradients to convert to vector")
    return torch.cat(l)


def grad_dict_norm(grad_dict, norm_type=2):
    return grad_dict_to_vector(grad_dict).norm(norm_type).item()
