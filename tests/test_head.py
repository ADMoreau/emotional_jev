"""PointerHead must start near chance whatever the backbone's hidden-state scale."""
import math

import torch

from facedecide.model import PointerHead


def test_head_is_scale_invariant_and_starts_near_chance():
    torch.manual_seed(0)
    head = PointerHead(2048, dp=256)
    h_decide, h_opts = torch.randn(2048), torch.randn(6, 2048)
    torch.testing.assert_close(head(1000 * h_decide, 1000 * h_opts), head(h_decide, h_opts), atol=1e-4, rtol=1e-4)
    # with large, uneven hidden states (like a real backbone's) the initial loss stays near ln(K)
    big = torch.randn(64, 7, 2048) * torch.logspace(0, 3, 2048)
    losses = [-torch.log_softmax(head(b[0], b[1:]), -1)[0] for b in big]
    assert abs(float(torch.stack(losses).mean()) - math.log(6)) < 0.3
