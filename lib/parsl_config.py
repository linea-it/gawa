from parsl import ThreadPoolExecutor
from parsl.config import Config
from parsl.providers import CondorProvider, SlurmProvider, LocalProvider
from parsl.executors import HighThroughputExecutor
from parsl.launchers import SrunLauncher
from parsl.addresses import address_by_hostname, address_by_interface
import os


def get_config(key):
    """
    Creates an instance of the Parsl configuration 

    Args:
        phz_config (dict): Photo-z pipeline configuration - available in the config.yml
    """

    gawa_root_dir = os.getenv('GAWA_ROOT', '.')

    executors = {
        "htcondor": HighThroughputExecutor(
            label='htcondor',
            address=address_by_hostname(),
            max_workers=54,
            worker_debug=True,
            provider=CondorProvider(
                init_blocks=15,
                min_blocks=15,
                max_blocks=16,
                parallelism=0.5,
                scheduler_options='+RequiresWholeMachine = True',
                worker_init=f"source {gawa_root_dir}/env.sh",
                cmd_timeout=120,
            ),
        ),
        "sdumont": HighThroughputExecutor(
            # address=address_by_interface('ib0'),
            address=address_by_hostname(),
            label='sd',
            max_workers=24, # number of cores per node           
            provider=SlurmProvider(
                partition='cpu_small',
                nodes_per_block=10, # number of nodes
                cmd_timeout=240, # duration for which the provider will wait for a command to be invoked on a remote system
                launcher=SrunLauncher(debug=True, overrides=''),
                init_blocks=5,
                min_blocks=5,
                max_blocks=5,
                parallelism=1,
                walltime='03:20:00',
                worker_init=f"source {gawa_root_dir}/env.sh\n"
            ),
        ),
        "local": HighThroughputExecutor(
            label='local',
            worker_debug=False,
            max_workers=4,
            provider=LocalProvider(
                min_blocks=1,
                init_blocks=1,
                max_blocks=1,
                parallelism=1,
                worker_init=f"source {gawa_root_dir}/env.sh\n",
            )
        ),
        "local_threads": ThreadPoolExecutor(
            label='local_threads',
            max_threads=2
        )
    }

    return Config(
        executors=[executors[key]],
        strategy=None
    )
