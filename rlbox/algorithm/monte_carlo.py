import numpy as np

from rlbox.core import Spec


def off_policy_monte_carlo(act_spec: Spec, obs_spec: Spec,
                           behavior_policy,
                           episodes, gamma=1.0):
    q = np.full(shape=(obs_spec.size(), act_spec.size()), fill_value=-10000,
                dtype=np.float64)
    c = np.zeros(shape=q.shape, dtype=np.float64)
    pi = np.random.choice(a=np.arange(act_spec.size()),
                          size=(obs_spec.size(),))

    for episode in episodes:
        g = 0.0
        w = 1.0
        for step in episode:
            s, a, r = step[0], step[1], step[2]
            if s == -1:
                break
            g = gamma * g + r
            c[s, a] += w
            q[s, a] += (w / c[s, a]) * (g - q[s, a])
            pi[s] = np.argmax(q[s, :])
            if pi[s] != a:
                break
            w = w / behavior_policy[s, a]
    return pi
