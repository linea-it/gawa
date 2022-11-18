from parsl.providers import SlurmProvider
from parsl.executors import HighThroughputExecutor
from parsl.addresses import address_by_hostname, address_by_interface
from parsl.launchers import SrunLauncher
import os


def create_executor(config):
    """create a new executor to SDumont

    Args:
        config (dict): Parsl configuration

    Returns:
        HighThroughputExecutor: Parsl executor
    """

    debug = True if os.getenv("GAWA_LOG_LEVEL", "") == "debug" else False

    provider = config.pop("provider_opts", {})
    provider["launcher"] = SrunLauncher(debug=debug, overrides="")
    provider["address"] = address_by_hostname()

    interface = config.pop("interface", None)

    if interface:
        provider["address"] = address_by_interface(interface)

    if not "worker_init" in provider:
        worker_init = f"source {os.getenv('GAWA_ROOT', '.')}/gawa.sh"
        provider["worker_init"] = worker_init

    config["provider"] = SlurmProvider(**provider)
    return HighThroughputExecutor(**config)
