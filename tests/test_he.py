import numpy as np
from dataloader import he_sampling_v2


def test_he_sampling_v2():
    adj = np.array([[0,1,0],
                    [1,0,1],
                    [0,1,1]])
    he = he_sampling_v2(adj, num_steps=3, max_num_he=2, eps_prob=0.001)
    assert len(he) == 2

