from parsl.providers import CondorProvider
from parsl.executors import HighThroughputExecutor
from parsl.addresses import address_by_hostname, address_by_interface
import os


class ICEXExecutor(HighThroughputExecutor):
    def __init__(self, **kwargs):
        provider = CondorProvider(
            init_blocks=1,
            min_blocks=1,
            max_blocks=1,
            parallelism=0.5,
            scheduler_options="+RequiresWholeMachine = True",
            worker_init=f"source {os.getenv('GAWA_ROOT', '.')}/gawa.sh",
            cmd_timeout=120,
        )
        super().__init__(provider=provider)
        self.__dict__.update(kwargs)

        if "interface" in kwargs:
            self.address = address_by_interface(kwargs["interface"])
        else:
            self.address = address_by_hostname()

        if "provider" in kwargs:
            self.provider.__dict__.update(kwargs["provider"])
