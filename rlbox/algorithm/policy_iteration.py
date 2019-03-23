import numpy as np

from rlbox.env.mdp import Mdp


def value(mdp: Mdp, state, action, V, gamma):
    """Bellman Equation for State Value

    Parameters
    ----------
    mdp : Mdp
        markov decision process instance
    state : vector
        vector with 1 set on the current state index and 0 on the others
    action : int
        chosen action in current state
    V : vector
        V(s) - vector with values for each state.
    gamma : float
        Discount factor.

    Returns
    -------
    out : float
        State value computed using Bellman Equation.
    """

    prob = np.dot(state, mdp.P[..., action])
    return np.sum(prob * (np.dot(state, mdp.R[..., action]) + gamma * V))


def policy_evaluation(mdp: Mdp, policy, V, gamma, epsilon=0.01):
    """Policy evaluation algorithm.

    Parameters
    ----------
    mdp : Mdp
        markov decision process instance
    policy : vector
        vector with action to choose on each state index
    V : vector
        V(s) - vector with values for each state, will be updated with state
        values for current policy
    gamma : float
        Discount factor.
    epsilon : float, optional
        stopping criteria small value, defaults to 0.01
    """

    while True:
        V0 = np.copy(V)
        for i, _ in enumerate(mdp.S):
            s = np.zeros(shape=mdp.S.shape)
            s[i] = 1.0
            V[i] = value(mdp, s, policy[i], V, gamma)
        delta = np.abs(V0 - V).max()  # stopping criteria
        if delta < epsilon:
            break


def policy_improvement(mdp: Mdp, policy, V, gamma):
    """Policy improvement algorithm.

    Parameters
    ----------
    mdp : Mdp
        markov decision process instance
    policy : vector
        vector with action to choose on each state index
    V : vector
        V(s) - vector with values for each state
    gamma : float
        Discount factor
    """

    for i, _ in enumerate(mdp.S):
        s = np.zeros(shape=mdp.S.shape)
        s[i] = 1.0
        action_returns = []
        for a, _ in enumerate(mdp.A):
            if np.sum(np.dot(s, mdp.P[..., a])):
                action_returns.append(value(mdp, s, a, V, gamma))
            else:
                action_returns.append(float('-inf'))
        policy[i] = np.argmax(action_returns)


def policy_iteration(mdp: Mdp, policy, gamma, epsilon=0.01):
    """Policy iteration algorithm.

    Parameters
    ----------
    mdp : Mdp
        markov decision process instance
    policy: vector
        initial policy to improve
    gamma : float
        Discount factor
    epsilon : float, optional
        stopping criteria small value, defaults to 0.01

    Returns
    -------
    policies: matrix
        matrix where each row is improved policy (action to choose on each
        state index) on each iteration, last policy is optimal
    values: vector
        V(s) - vector with values for each state
    """

    V = np.zeros(shape=mdp.S.shape)
    policies = [policy]

    while True:
        policy_evaluation(mdp, policy, V, gamma, epsilon)

        new_policy = np.copy(policy)

        policy_improvement(mdp, new_policy, V, gamma)

        policy_change = (new_policy != policy).sum()
        policy = new_policy
        if policy_change == 0:
            break
        else:
            policies.append(new_policy)
    return np.array(policies), V


def value_iteration(mdp: Mdp, gamma, epsilon=0.01):
    """Value iteration algorithm.

    Parameters
    ----------
    mdp : Mdp
        markov decision process instance
    gamma : float
        Discount factor
    epsilon : float, optional
        stopping criteria small value, defaults to 0.01

    Returns
    -------
    policy: vector
        optimal action to choose on each state index
    values: matrix
        matrix where each row is improved value vector on each iteration,
        last row is optimal
    """

    V = np.zeros(shape=mdp.S.shape)
    policy = np.zeros(shape=mdp.S.shape, dtype=np.int32)
    values = []

    while True:
        V0 = np.copy(V)
        for i, _ in enumerate(mdp.S):
            s = np.zeros(shape=mdp.S.shape)
            s[i] = 1.0
            action_returns = []
            for a, _ in enumerate(mdp.A):
                if not np.sum(mdp.P[i, :, a]):
                    continue
                if np.sum(np.dot(s, mdp.P[..., a])):
                    action_returns.append(value(mdp, s, a, V, gamma))
                else:
                    action_returns.append(float('-inf'))
            if action_returns:
                V[i] = np.max(action_returns)
                policy[i] = np.argmax(np.round(action_returns, 5))

        delta = np.abs(V0 - V).max()  # stopping criteria
        if delta < epsilon:
            break
        else:
            values.append(V.copy())

    return policy, np.array(values)
