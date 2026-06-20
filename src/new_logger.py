import netket as nk
from netket.utils import struct

from nqxpack import load as nqx_load
from nqxpack import save as nqx_save

import jax
from typing import Tuple, Any, IO, TYPE_CHECKING
import numpy as np
from pathlib import Path

import orjson

from netket.utils.history import accum_histories_in_tree, HistoryDict
from netket.logging.base import AbstractLog

class CheckpointCallback():
    """Periodically saves a resumable snapshot (state + log).

    Fires at ``on_step_end`` (after ``update_parameters``), so the saved
    parameters and time are post-update.
    """

    output_dir: str = struct.field(pytree_node=False)
    logger: object = struct.field(pytree_node=False)
    every_n_steps: int = struct.field(pytree_node=False)
    wandb_run: None

    def __init__(self, output_dir, logger, every_n_steps, wandb_run=None):
        self.output_dir = output_dir
        self.logger = logger
        self.every_n_steps = every_n_steps
        self.wandb_run = wandb_run

    def on_step_end(self, step, log_data, driver):
        if step % self.every_n_steps != 0:
            return

        # Save metadata (t, step) so restore doesn't depend on dt
        if jax.process_index() == 0:
            self.logger.serialize(self.output_dir + "/log")
            nqx_save(driver.state, self.output_dir + f"/state.nk")
            if self.wandb_run is not None:
                data = self.logger.data.to_dict()
                if data:
                    self.wandb_run.log(_clean(data))
            print(f"  Checkpoint at step {step}")

    def __call__(self, step, log_data, driver):
        self.on_step_end(step, log_data, driver)
        return True

    def restore_state(self) -> nk.vqs.VariationalState | None:
        try:
            return nqx_load(self.output_dir + f"/state.nk")
        except FileNotFoundError:
            print(f"No checkpoint found in {self.output_dir}")
            return None

    def restore_logger(self) -> Tuple[nk.logging.RuntimeLog, bool]:
        try:
            self.logger = RuntimeLog.deserialize(self.output_dir + f"/log")
        except FileNotFoundError:
            print(f"No logger found in {self.output_dir}")
            return self.logger, False
        # step = logger.data["step"][-1]
        return self.logger, True

    def finish(self, state):
        if jax.process_index() == 0:
            self.logger.serialize(self.output_dir + "/log")
            nqx_save(state, self.output_dir + f"/state.nk")
            if self.wandb_run is not None:
                data = self.logger.data.to_dict()
                if data:
                    self.wandb_run.log(_clean(data))


def _clean(data):
    new_data = {}
    for k_main, history in data.items():
        for k, v in history.to_dict().items():
            if k == "iters":
                continue
            if np.iscomplexobj(v[0]):
                new_data[f"{k_main}.{k}Re"] = v.real[-1]
                new_data[f"{k_main}.{k}Im"] = v.imag[-1]
            else:
                new_data[f"{k_main}.{k}"] = v[-1]
    return new_data


class RuntimeLog(AbstractLog):
    """
    This logger accumulates log data in a set of nested dictionaries which are stored in memory. The log data is not automatically saved to the filesystem.

    It can be passed with keyword argument `out` to Monte Carlo drivers in order
    to serialize the output data of the simulation.

    This logger keeps the data in memory, and does not save it to disk. To serialize
    the current content to a file, use the method :py:meth:`~netket.logging.RuntimeLog.serialize`.
    """

    _data: dict[str, Any]

    def __init__(self):
        """
        Crates a Runtime Logger.
        """
        self._data: dict[str, Any] = HistoryDict()
        self._old_step = 0

    def __call__(
        self,
        step: int,
        item: dict[str, Any],
        variational_state: "VariationalState | None" = None,
    ):
        if self._data is None:
            self._data = {}
        self._data = accum_histories_in_tree(self._data, item, step=step)
        self._old_step = step

    @property
    def data(self) -> dict[str, Any]:
        """
        The dictionary of logged data.
        """
        return self._data

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def flush(self, variational_state=None):
        pass

    @classmethod
    def deserialize(cls, path: str | Path) -> "RuntimeLog":
        r"""
        Load a :class:`~netket.logging.RuntimeLog` from a file previously serialized with
        :py:meth:`~netket.logging.RuntimeLog.serialize`.

        If the path has no extension, ``.json`` is appended, consistent with the
        behaviour of :py:meth:`~netket.logging.RuntimeLog.serialize`.

        Args:
            path: The path of the file to load.

        Returns:
            A new :class:`~netket.logging.RuntimeLog` instance with the data from the file.
        """
        if isinstance(path, str):
            path = Path(path)

        if isinstance(path, Path):
            filename = path.name
            if not filename.endswith((".log", ".json")):
                path = path.parent / (filename + ".json")

        data = HistoryDict.from_file(path)

        log = cls.__new__(cls)
        log._data = data

        # Recover _old_step from the maximum last iter across all histories
        def _find_last_step(d):
            last = 0
            for val in d.values():
                if isinstance(val, dict):
                    last = max(last, _find_last_step(val))
                elif hasattr(val, "iters") and len(val.iters) > 0:
                    last = max(last, int(val.iters[-1]))
            return last

        log._old_step = _find_last_step(data._data)

        return log

    def serialize(self, path: str | Path | IO):
        r"""
        Serialize the content of :py:attr:`~netket.logging.RuntimeLog.data` to a file.

        If the file already exists, it is overwritten.

        Args:
            path: The path of the output file. It must be a valid path.
        """
        if isinstance(path, str):
            path = Path(path)

        if not self._is_master_process:
            return

        if isinstance(path, Path):
            parent = path.parent
            filename = path.name
            if not filename.endswith((".log", ".json")):
                filename = filename + ".json"
            path = parent / filename

            with open(path, "wb") as io:
                self._serialize(io)
        else:
            self._serialize(path)

    def _serialize(self, outstream: IO):
        r"""
        Inner method of `serialize`, working on an IO object.
        """
        outstream.write(
            orjson.dumps(
                self.data,
                default=default,
                option=orjson.OPT_SERIALIZE_NUMPY,
            )
        )

    def __repr__(self):
        _str = "RuntimeLog():\n"
        if self.data is not None:
            _str += f" keys = {list(self.data.keys())}"
        return _str


def default(obj):
    if hasattr(obj, "to_json"):
        return obj.to_json()
    elif hasattr(obj, "to_dict"):
        return obj.to_dict()
    elif isinstance(obj, np.ndarray):
        if np.issubdtype(obj.dtype, np.complexfloating):
            return {
                "real": np.ascontiguousarray(obj.real),
                "imag": np.ascontiguousarray(obj.imag),
            }
        else:
            return np.ascontiguousarray(obj)
    elif isinstance(obj, jax.numpy.ndarray):
        return np.ascontiguousarray(obj)
    elif isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}

    raise TypeError
