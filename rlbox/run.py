from absl import app, flags

import rlbox
from rlbox.testbed.config import Algorithm as alg

flags.DEFINE_enum('testbed', None,
                  [
                      'narmedbandit.SampleAverage',
                      'narmedbandit.WeightedAverage',
                      'narmedbandit.OptInitVal',
                      'narmedbandit.Ucb',
                      'narmedbandit.Gradient',
                      'narmedbandit.ParamStudy',
                      'car_rental_v1',
                      'car_rental_v2',
                      'gambler.0.4',
                      'gambler.0.25',
                      'gambler.0.55',
                      'racetrack',
                      'gridworld.windy',
                      'gridworld.windy_stochastic',
                      'gridworld.NStepSarsa',
                      'maze.DynaQ',
                      'maze.DynaQ+',
                      'mountain_car.SemiGradientSarsa',
                      'mountain_car.TrueSarsaLambda',
                      'mountain_car.ActorCritic'
                  ],
                  'Name of a testbed that you want to use.')
flags.DEFINE_bool('start', True, 'Run experiment using a chosen testbed.')
flags.DEFINE_bool('plot', True,
                  'Plot data that was generated with a chosen testbed.')

flags.mark_flag_as_required('testbed')


def main(argv):
    import time

    args = flags.FLAGS

    if args.testbed == 'narmedbandit.SampleAverage':
        testbed = rlbox.testbed.narmedbandit.NArmedBanditTestbed(
            runs=2000,
            env={'steps': 1000, 'arms': 10, 'stationary': True},
            exe=[
                (alg.SMPL_AVG, {'epsilon': 0.0}),
                (alg.SMPL_AVG, {'epsilon': 0.01}),
                (alg.SMPL_AVG, {'epsilon': 0.1})
            ])

    if args.testbed == 'narmedbandit.WeightedAverage':
        testbed = rlbox.testbed.narmedbandit.NArmedBanditTestbed(
            runs=2000,
            env={'steps': 10000, 'arms': 10, 'stationary': False},
            exe=[
                (alg.SMPL_AVG, {'epsilon': 0.1}),
                (alg.WEIGHT_AVG, {'epsilon': 0.1, 'alpha': 0.2})
            ])

    if args.testbed == 'narmedbandit.OptInitVal':
        testbed = rlbox.testbed.narmedbandit.NArmedBanditTestbed(
            runs=2000,
            env={'steps': 1000, 'arms': 10, 'stationary': True},
            exe=[
                (alg.WEIGHT_AVG, {'epsilon': 0.0, 'alpha': 0.1, 'bias': 5.0}),
                (alg.WEIGHT_AVG, {'epsilon': 0.1, 'alpha': 0.1, 'bias': 0.0})
            ])

    if args.testbed == 'narmedbandit.Ucb':
        testbed = rlbox.testbed.narmedbandit.NArmedBanditTestbed(
            runs=2000,
            env={'steps': 1000, 'arms': 10, 'stationary': True},
            exe=[
                (alg.SMPL_AVG, {'epsilon': 0.1}),
                (alg.UCB, {'c': 2}),
            ])

    if args.testbed == 'narmedbandit.Gradient':
        testbed = rlbox.testbed.narmedbandit.NArmedBanditTestbed(
            runs=2000,
            env={'steps': 1000, 'arms': 10, 'stationary': True, 'mean': 4.0},
            exe=[
                (alg.GRADIENT, {'alpha': 0.1, 'baseline': True}),
                (alg.GRADIENT, {'alpha': 0.4, 'baseline': True}),
                (alg.GRADIENT, {'alpha': 0.1, 'baseline': False}),
                (alg.GRADIENT, {'alpha': 0.4, 'baseline': False})
            ])

    if args.testbed == 'narmedbandit.ParamStudy':
        testbed = rlbox.testbed.narmedbandit.NArmedBanditParamStudy(
            runs=2000,
            env={'steps': 1000, 'arms': 10, 'stationary': True},
            exe=[
                (alg.SMPL_AVG, {'epsilon': 1 / 128}),
                (alg.SMPL_AVG, {'epsilon': 1 / 64}),
                (alg.SMPL_AVG, {'epsilon': 1 / 32}),
                (alg.SMPL_AVG, {'epsilon': 1 / 16}),
                (alg.SMPL_AVG, {'epsilon': 1 / 8}),
                (alg.SMPL_AVG, {'epsilon': 1 / 4}),

                (alg.GRADIENT, {'alpha': 1 / 32}),
                (alg.GRADIENT, {'alpha': 1 / 16}),
                (alg.GRADIENT, {'alpha': 1 / 8}),
                (alg.GRADIENT, {'alpha': 1 / 4}),
                (alg.GRADIENT, {'alpha': 1 / 2}),
                (alg.GRADIENT, {'alpha': 1}),
                (alg.GRADIENT, {'alpha': 2}),
                (alg.GRADIENT, {'alpha': 4}),

                (alg.UCB, {'c': 1 / 16}),
                (alg.UCB, {'c': 1 / 8}),
                (alg.UCB, {'c': 1 / 4}),
                (alg.UCB, {'c': 1 / 2}),
                (alg.UCB, {'c': 1}),
                (alg.UCB, {'c': 2}),
                (alg.UCB, {'c': 4}),

                (alg.WEIGHT_AVG, {'epsilon': 0.0, 'alpha': 0.1, 'bias': 1 / 4}),
                (alg.WEIGHT_AVG, {'epsilon': 0.0, 'alpha': 0.1, 'bias': 1 / 2}),
                (alg.WEIGHT_AVG, {'epsilon': 0.0, 'alpha': 0.1, 'bias': 1}),
                (alg.WEIGHT_AVG, {'epsilon': 0.0, 'alpha': 0.1, 'bias': 2}),
                (alg.WEIGHT_AVG, {'epsilon': 0.0, 'alpha': 0.1, 'bias': 4})
            ])

    if args.testbed == 'car_rental_v1':
        testbed = rlbox.testbed.car_rental.CarRentalTestbed(
            max_move=5, max_cars=20, expct=[3, 4, 3, 2], s0=[0, 0],
            gamma=0.9, epsilon=1.0)

    if args.testbed == 'car_rental_v2':
        testbed = rlbox.testbed.car_rental.CarRentalTestbed(
            max_move=5, max_cars=20, expct=[3, 4, 3, 2], s0=[0, 0],
            gamma=0.9, epsilon=1.0, modified=True)

    if args.testbed == 'gambler.0.4':
        testbed = rlbox.testbed.gambler.GamblerTestbed(
            0.4, 100, gamma=1.0, epsilon=1e-9)

    if args.testbed == 'gambler.0.25':
        testbed = rlbox.testbed.gambler.GamblerTestbed(
            0.25, 100, gamma=1.0, epsilon=1e-9)

    if args.testbed == 'gambler.0.55':
        testbed = rlbox.testbed.gambler.GamblerTestbed(
            0.55, 100, gamma=1.0, epsilon=0.01)

    if args.testbed == 'racetrack':
        testbed = rlbox.testbed.racetrack.RaceTrackTestbed(
            runs=50000,
            env={'steps': 10000},
            exe=[
                (alg.RANDOM, {}),
            ])

    if args.testbed == 'gridworld.windy':
        testbed = rlbox.testbed.windy_gridworld.WindyGridWorldTestbed(
            runs=200, stochastic=False, alpha=0.5, epsilon=0.1,
            gamma=1.0)

    if args.testbed == 'gridworld.windy_stochastic':
        testbed = rlbox.testbed.windy_gridworld.WindyGridWorldTestbed(
            runs=200, stochastic=True, alpha=0.5, epsilon=0.1,
            gamma=1.0)

    if args.testbed == 'gridworld.NStepSarsa':
        testbed = rlbox.testbed.nstep_sarsa.NStepSarsaTestbed(
            runs=200, stochastic=False, n=3, alpha=0.5, epsilon=0.1,
            gamma=1.0)

    if args.testbed == 'maze.DynaQ':
        testbed = rlbox.testbed.maze.MazeTestbed(
            runs=30,
            env={'maze_type': 0},
            exe=[
                (alg.DYNA_Q, {'n': 50, 'alpha': 0.1, 'epsilon': 0.1,
                              'gamma': 0.95}, {'episodes': 50}),
                (alg.DYNA_Q, {'n': 5, 'alpha': 0.1, 'epsilon': 0.1,
                              'gamma': 0.95}, {'episodes': 50}),
                (alg.DYNA_Q, {'n': 0, 'alpha': 0.1, 'epsilon': 0.1,
                              'gamma': 0.95}, {'episodes': 50})
            ])

    if args.testbed == 'maze.DynaQ+':
        testbed = rlbox.testbed.maze.BlockingMazeTestbed(
            runs=30,
            env={'maze_type': 1},
            exe=[
                (alg.DYNA_Q,
                 {'n': 10, 'alpha': 1.0, 'epsilon': 0.1, 'gamma': 0.95},
                 {'steps': 3000}),
                (alg.DYNA_Q,
                 {'n': 10, 'alpha': 1.0, 'epsilon': 0.1, 'gamma': 0.95,
                  'kappa': 1e-4},
                 {'steps': 3000}),
                (alg.DYNA_Q_V2,
                 {'n': 10, 'alpha': 1.0, 'epsilon': 0.1, 'gamma': 0.95,
                  'kappa': 1e-4},
                 {'steps': 3000})
            ])

    if args.testbed == 'mountain_car.SemiGradientSarsa':
        testbed = rlbox.testbed.mountain_car.MountainCarTestbed(
            runs=10,
            env={},
            exe=[
                (alg.SEMIGRADIENT_SARSA, {'alpha': 0.5, 'epsilon': 0.0,
                                          'gamma': 1.0}, {'episodes': 500})
            ])

    if args.testbed == 'mountain_car.TrueSarsaLambda':
        testbed = rlbox.testbed.mountain_car.MountainCarTestbed(
            runs=10,
            env={},
            exe=[
                (alg.TRUE_SARSA_LAMBDA, {'alpha': 0.5, 'epsilon': 0.0,
                                         'gamma': 1.0, 'lmbda': 0.9},
                 {'episodes': 500})
            ])

    if args.testbed == 'mountain_car.ActorCritic':
        testbed = rlbox.testbed.mountain_car.MountainCarTestbed(
            runs=10,
            env={},
            exe=[
                (alg.ACTOR_CRITIC, {'alpha_w': 0.2, 'alpha_theta': 0.01,
                                    'gamma': 1.0, 'lambda_w': 0.9,
                                    'lambda_theta': 0.9},
                 {'episodes': 500})
            ])

    if args.start:
        start = time.perf_counter()
        testbed.run()
        print('TIME: ' + str(time.perf_counter() - start) + ' [s]')

    if args.plot:
        testbed.plot()


if __name__ == '__main__':
    app.run(main)
