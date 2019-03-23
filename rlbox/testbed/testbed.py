import multiprocessing as mp
import os
from contextlib import contextmanager

import tqdm

from .sink import Sink


@contextmanager
def testbed_executor(sink_cfg, summary):
    with Sink(sink_cfg, summary) as data:
        with mp.Pool(mp.cpu_count(),
                     initializer=data.share_with,
                     initargs=data.to_share()) as pool:
            yield pool


class Testbed:
    """Multiprocess experimental runs of the agents in an environment with data
    gathering and plotting."""

    def __init__(self, runs, env, exe, summary):
        """
        Parameters
        ----------
        runs: int
            number of runs of each task
        env: dict
            environment configuration
        exe: list of tuples
            configuration of tasks to execute
        summary: str
            name of hdf5 file with data collected from experiment
        """

        dump_dir = r'.dump'
        if not os.path.exists(dump_dir):
            os.mkdir(dump_dir)
        self.summary = r'{0}/{1}.h5'.format(dump_dir, summary)
        self.exe = exe
        self.runs = runs
        self.env = env

    def run(self):
        with testbed_executor(self.sink_cfg(), self.summary) as pool:
            tasks = [(self.runs, self.env, e, {'run': run})
                     for e in self.exe
                     for run in range(self.runs)]
            for _ in tqdm.tqdm(pool.imap(self._worker, tasks),
                               total=len(tasks)):
                pass

    def _worker(self, arg):
        return self.worker(*arg)

    @staticmethod
    def worker(runs, env_cfg, task, param):
        """
        Parameters
        ----------
        runs: int
            number of runs of each task
        env_cfg: dict
             environment configuration
        task: tuple
            configuration of task to execute
        param: dict
            additional parameters, etc. current run number
        """
        pass

    @staticmethod
    def key_for(kind, **kwargs):
        key_name = '/'.join([key + '_' + str(kwargs[key]) for key in kwargs])
        key_name += '/' + kind
        return key_name.replace('.', '_')

    def plot(self):
        pass

    def sink_cfg(self):
        pass
