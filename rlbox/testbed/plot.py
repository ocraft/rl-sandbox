import numpy as np
from matplotlib.ticker import FuncFormatter

from rlbox.env.narmedbandit import NArmedBanditEnv


def labels_for(**kwargs):
    args = [label_for(key) + str(kwargs[key]) for key in kwargs]
    return '[' + ', '.join(args) + ']'


def label_for(key):
    if key == 'alg':
        return ''

    return prettify(key) + '='


def prettify(key):
    key_lbl = key
    if key == 'epsilon':
        key_lbl = r'$\varepsilon$'
    if key == 'alpha':
        key_lbl = r'$\alpha$'
    if key == 'alpha_w':
        key_lbl = r'$\alpha^w$'
    if key == 'alpha_theta':
        key_lbl = r'$\alpha^{\theta}$'
    if key == 'gamma':
        key_lbl = r'$\gamma$'
    if key == 'kappa':
        key_lbl = r'$\kappa$'
    if key == 'lmbda':
        key_lbl = r'$\lambda$'
    if key == 'lambda_w':
        key_lbl = r'$\lambda^w$'
    if key == 'lambda_theta':
        key_lbl = r'$\lambda^{\theta}$'

    return key_lbl


def average_reward_plt(axis, label):
    axis.set_xlabel('Steps')
    axis.set_ylabel('Average reward')
    axis.legend(loc='best', labels=label)
    axis.autoscale()


def optimal_action_plt(axis, label):
    axis.set_xlabel('Steps')
    axis.set_ylabel('% Optimal action')
    axis.yaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: '{:2.0f}%'.format(x * 100)))
    axis.legend(loc='best', labels=label)
    axis.autoscale()


def bandit_problem_plt(axis, **env):
    narmedenv = NArmedBanditEnv(**env)
    axis.set_title('An example bandit problem from the {0}-armed testbed.'
                   .format(env['arms']))
    axis.set_xlabel('Action')
    axis.set_ylabel('Reward distribution')
    axis.violinplot([np.random.normal(loc=m, size=10000)
                     for m in narmedenv.bandit.qstar_means], showmeans=True)
    axis.set_xticks(range(1, env['arms'] + 1, 1))
    qstar_style = dict(size=8, color='black')
    qstar_lbl = r'$q_*({0})$'
    for arm, m in enumerate(narmedenv.bandit.qstar_means, 1):
        axis.annotate(qstar_lbl.format(arm), xy=(arm, m),
                      xytext=(arm + 0.15, m), **qstar_style)
