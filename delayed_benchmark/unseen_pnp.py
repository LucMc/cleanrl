import metaworld
import pickle
from collections import OrderedDict
from typing import Any

import numpy as np
import numpy.typing as npt

import metaworld.envs.mujoco.env_dict as _env_dict
from metaworld.types import Task
from cleanrl_utils.wrappers.wrappers_rd import UnseenRandomDelayWrapper, RandomDelayWrapper, AugmentedRandomDelayWrapper, NoneWrapper

# class MT1(Benchmark):
#     """The MT1 benchmark. A goal-conditioned RL environment for a single Metaworld task."""
#
#     ENV_NAMES = list(_env_dict.ALL_V2_ENVIRONMENTS.keys())
#
#     def __init__(self, env_name, seed=None):
#         super().__init__()
#         if env_name not in _env_dict.ALL_V2_ENVIRONMENTS:
#             raise ValueError(f"{env_name} is not a V2 environment")
#         cls = _env_dict.ALL_V2_ENVIRONMENTS[env_name]
#         self._train_classes = OrderedDict([(env_name, cls)])
#         self._test_classes = OrderedDict([(env_name, cls)])
#         args_kwargs = _env_dict.ML1_args_kwargs[env_name]
#
#         self._train_tasks = _make_tasks(
#             self._train_classes, {env_name: args_kwargs}, _MT_OVERRIDE, seed=seed
#         )
#
#         self._test_tasks = []
#
# class MT10(Benchmark):
#     """The MT10 benchmark. Contains 10 tasks in its train set. Has an empty test set."""
#
#     def __init__(self, seed=None):
#         super().__init__()
#         self._train_classes = _env_dict.MT10_V2
#         self._test_classes = OrderedDict()
#         train_kwargs = _env_dict.MT10_V2_ARGS_KWARGS
#         self._train_tasks = _make_tasks(
#             self._train_classes, train_kwargs, _MT_OVERRIDE, seed=seed
#         )
#
#         self._test_tasks = []
#         self._test_classes = []

def _make_delayed_tasks(
    classes: _env_dict.EnvDict,
    args_kwargs: _env_dict.EnvArgsKwargsDict,
    kwargs_override: dict,
    seed: int | None = None,
    delay_info: Dict = None # Include default here?
) -> list[Task]:
    """Initialises goals for a given set of environments.

    Args:
        classes: The environment classes as an `EnvDict`.
        args_kwargs: The environment arguments and keyword arguments.
        kwargs_override: Any kwarg overrides.
        seed: The random seed to use.

    Returns:
        A flat list of `Task` objects, `_N_GOALS` for each environment in `classes`.
    """
    # Cache existing random state
    if seed is not None:
        st0 = np.random.get_state()
        np.random.seed(seed)

    tasks = []
    for env_name, args in args_kwargs.items():
        kwargs = args["kwargs"].copy()
        assert isinstance(kwargs, dict)
        assert len(args["args"]) == 0

        # Init env
        env = classes[env_name]()
        env._freeze_rand_vec = False
        env._set_task_called = True
        rand_vecs: list[npt.NDArray[Any]] = []

        # Set task
        del kwargs["task_id"]
        env._set_task_inner(**kwargs)

        for _ in range(_N_GOALS):  # Generate random goals
            env.reset()
            assert env._last_rand_vec is not None
            rand_vecs.append(env._last_rand_vec)

        unique_task_rand_vecs = np.unique(np.array(rand_vecs), axis=0)
        assert (
            unique_task_rand_vecs.shape[0] == _N_GOALS
        ), f"Only generated {unique_task_rand_vecs.shape[0]} unique goals, not {_N_GOALS}"
        env.close()

        # Create a task for each random goal
        for rand_vec in rand_vecs:
            kwargs = args["kwargs"].copy()
            assert isinstance(kwargs, dict)
            del kwargs["task_id"]

            kwargs.update(dict(rand_vec=rand_vec, env_cls=classes[env_name]))
            kwargs.update(kwargs_override)

            tasks.append(_encode_task(env_name, kwargs))

        del env

    # Restore random state
    if seed is not None:
        np.random.set_state(st0)

    return tasks


class UnseenBench(Benchmark):

    ENV_NAMES = list(_env_dict.ALL_V2_ENVIRONMENTS.keys())

    """ delay_info : {delay_type: gym.Wrapper,
                      min_obs_delay: int,
                      ...
                      max_act_delay: int}
    """
    def __init__(self, env_name, seed=None, delay_info: OrderedDict):
        super().__init__()

        if env_name not in _env_dict.ALL_V2_ENVIRONMENTS:
            raise ValueError(f"{env_name} is not a V2 environment")

        cls = _env_dict.ALL_V2_ENVIRONMENTS[env_name]
        self._train_classes = OrderedDict([(env_name, cls)])
        self._test_classes = OrderedDict([(env_name, cls)])
        args_kwargs = _env_dict.ML1_args_kwargs[env_name]

        self._train_tasks = _make_delayed_tasks(
            self._train_classes,
            {env_name: args_kwargs},
            _MT_OVERRIDE,
            seed=seed,
            delay_info=delay_info
        )

        self._test_tasks = []


if __name__ == "__main__":
    # Test benchmark works
    benchmark = metaworld.MT1('pick-place-v2')


