from parsl.config import Config


def get_parsl_config(config) -> Config:
    executors, executor = config.get("executors"), config.get("executor")
    executor_dict = dict(executors.get(executor))
    mod = __import__(f"executors.{executor}", fromlist=["create_executor"])
    _func = getattr(mod, "create_executor")
    executor_instance = _func(executor_dict)
    return Config(executors=[executor_instance], strategy=None)
