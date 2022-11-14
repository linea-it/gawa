from parsl.providers import SlurmProvider
from parsl.executors import HighThroughputExecutor
from parsl.addresses import address_by_hostname, address_by_interface
from parsl.launchers import SrunLauncher
import os


class SDExecutor(HighThroughputExecutor):
    def __init__(self, **kwargs):
        debug = True if os.getenv("GAWA_LOG_LEVEL", "") == "debug" else False

        provider = SlurmProvider(
            partition="cpu_dev",
            nodes_per_block=2,  # number of nodes
            cmd_timeout=240,  # duration for which the provider will wait for a command to be invoked on a remote system
            launcher=SrunLauncher(debug=debug, overrides=""),
            init_blocks=1,
            min_blocks=1,
            max_blocks=1,
            parallelism=0.5,
            walltime="00:20:00",
            worker_init=f"source {os.getenv('GAWA_ROOT', '.')}/gawa.sh",
        )
        super().__init__(provider=provider)
        self.__dict__.update(kwargs)

        if "interface" in kwargs:
            self.address = address_by_interface(kwargs["interface"])
        else:
            self.address = address_by_hostname()

        if "provider" in kwargs:
            self.provider.__dict__.update(kwargs["provider"])
