import rlbox.core as core
from rlbox.agent.approx.linear import SemiGradientSarsa, TrueOnlineSarsaLambda
from rlbox.agent.approx.linear import ActorCriticLambda
from rlbox.env.mountain_car import MountainCar


def test_runs_in_mountain_car_environment():
    environment = MountainCar()
    agent_program = SemiGradientSarsa(
        environment.act_spec,
        environment.obs_spec,
        alpha=0.5,
        epsilon=0.0,
        gamma=1.0)
    agent = core.Agent(
        agent_program,
        lambda env: (env.obs, env.reward, env.done()),
        lambda action, env: env.step(action))
    core.Run(agent, environment).start()


def test_true_sarsa_lambda_runs_in_mountain_car_environment():
    environment = MountainCar()
    agent_program = TrueOnlineSarsaLambda(
        environment.act_spec,
        environment.obs_spec,
        alpha=0.2,
        epsilon=0.0,
        gamma=1.0,
        lmbda=0.9)
    agent = core.Agent(
        agent_program,
        lambda env: (env.obs, env.reward, env.done()),
        lambda action, env: env.step(action))
    core.Run(agent, environment).start()


def test_actor_critic_runs_in_mountain_car_environment():
    environment = MountainCar()
    agent_program = ActorCriticLambda(
        environment.act_spec,
        environment.obs_spec,
        alpha_w=0.1,
        alpha_theta=0.01,
        gamma=1.0,
        lambda_w=0.9,
        lambda_theta=0.9)
    agent = core.Agent(
        agent_program,
        lambda env: (env.obs, env.reward, env.done()),
        lambda action, env: env.step(action))
    core.Run(agent, environment).start()
