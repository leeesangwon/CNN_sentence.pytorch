def is_cuda(module):
    return next(module.parameters()).is_cuda
