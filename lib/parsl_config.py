from parsl.config import Config


def get_parsl_config(config) -> Config:
    executors, executor = config.get("executors"), config.get("executor")
    executor_dict = executors.get(executor)
    mod = __import__(f"executors.{executor}", fromlist=[executor_dict.get("classname")])
    cl = getattr(mod, executor_dict.get("classname"))
    executor_instance = cl(**executor_dict)
    return Config(executors=[executor_instance], strategy=None)
