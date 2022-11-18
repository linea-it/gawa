from parsl.providers import CondorProvider
from parsl.executors import HighThroughputExecutor
from parsl.addresses import address_by_hostname, address_by_interface
import os


def create_executor(config):
    """create a new executor to ICEX LIneA

    Args:
        config (dict): Parsl configuration

    Returns:
        HighThroughputExecutor: Parsl executor
    """

    provider = config.pop("provider_opts", {})
    provider["address"] = address_by_hostname()

    interface = config.pop("interface", None)

    if interface:
        provider["address"] = address_by_interface(interface)

    if not "worker_init" in provider:
        worker_init = f"source {os.getenv('GAWA_ROOT', '.')}/gawa.sh"
        provider["worker_init"] = worker_init

    config["provider"] = CondorProvider(**provider)
    return HighThroughputExecutor(**config)
