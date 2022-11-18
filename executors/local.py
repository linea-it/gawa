from parsl.providers import LocalProvider
from parsl.executors import HighThroughputExecutor
import os


def create_executor(config):
    """create a new executor to local execution

    Args:
        config (dict): Parsl configuration

    Returns:
        HighThroughputExecutor: Parsl executor
    """

    provider_opts = config.pop("provider_opts", {})

    if not "worker_init" in provider_opts:
        worker_init = f"source {os.getenv('GAWA_ROOT', '.')}/gawa.sh"
        provider_opts["worker_init"] = worker_init

    config["provider"] = LocalProvider(**provider_opts)
    return HighThroughputExecutor(**config)
