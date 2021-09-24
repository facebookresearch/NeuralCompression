import torch.nn


class LPIPS(torch.nn.Module):
    def __init__(self):
        super(LPIPS, self).__init__()
