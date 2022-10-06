from parsl.providers import LocalProvider
from parsl import ThreadPoolExecutor
from parsl.executors import HighThroughputExecutor
import os


class HTLocalExecutor(HighThroughputExecutor):

    def __init__(self, **kwargs):
        provider=LocalProvider(
            min_blocks=1,
            init_blocks=1,
            max_blocks=1,
            parallelism=1,
            worker_init = f"source {os.getenv('GAWA_ROOT', '.')}/gawa.sh",
        )
        super().__init__(provider=provider)
        self.__dict__.update(kwargs)


        if 'local_provider' in kwargs:
            self.provider.__dict__.update(kwargs['local_provider'])

