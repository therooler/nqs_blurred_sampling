from __future__ import annotations
from dataclasses import dataclass, field, asdict, replace, is_dataclass
from typing import Optional, Union, Literal, Dict, Any, Tuple
import os, yaml

# --- Optional imports used in builders ---
import jax
import jax.numpy as jnp
import netket as nk
from jax.nn.initializers import normal
import optax


# =========================
# Schedules (learning rate / diag_shift)
# =========================
@dataclass
class ConstantSchedule:
    value: float = 0.01

    def __post_init__(self):
        self.name = f"constant_{self.value:1.4f}"

    def build(self):
        return lambda s: self.value


@dataclass
class ExponentialDecaySchedule:
    init_value: float = 1e1
    decay_rate: float = 0.95
    end_value: float = 1e-4

    def __post_init__(self):
        self.name = f"exponential_init_{self.init_value:1.4f}_decay_{self.decay_rate:1.4f}_end_{self.end_value:1.6f}"

    def build(self):
        return optax.schedules.exponential_decay(
            init_value=self.init_value,
            transition_steps=1,
            decay_rate=self.decay_rate,
            end_value=self.end_value,
        )


@dataclass
class CosineDecaySchedule:
    init_value: float = 1e1
    decay_steps: int = 500
    end_value: float = 1e-4

    def __post_init__(self):
        self.name = f"cosine_init_{self.init_value:1.4f}_decay_steps_{self.decay_steps}_end_{self.end_value:1.6f}"

    def build(self):
        return optax.schedules.cosine_decay_schedule(
            init_value=self.init_value,
            decay_steps=self.decay_steps,
            alpha=self.end_value,
        )


Schedule = Union[ConstantSchedule, ExponentialDecaySchedule, CosineDecaySchedule, float]


def _build_schedule(s: Schedule):
    if isinstance(s, (ConstantSchedule, ExponentialDecaySchedule, CosineDecaySchedule)):
        return s.build()
    # float
    return s


# =========================
# Lattice / Hilbert / Hamiltonian / Sampler / Model
# =========================
@dataclass
class LatticeConfig:
    name = "AbstractLattice"
    _built_object: Optional[object] = field(default=None, init=False, repr=False)

    def build(self):
        raise NotImplementedError(f"Lattice type {self.name} not implemented")


@dataclass
class HypercubeConfig(LatticeConfig):
    type: Literal["Hypercube"] = "Hypercube"
    L: int = 10
    n_dim: int = 2
    pbc: bool = True
    max_neighbor_order: int = 2

    def __post_init__(self):
        self.name = f"HyperCube_{self.n_dim}D_L{self.L}_{'pbc' if self.pbc else ''}_nn_{self.max_neighbor_order}"

    def build(self):
        if self._built_object is not None:
            return self._built_object
        assert self.type == "Hypercube", "Only Hypercube is supported here"
        lattice = nk.graph.Hypercube(
            length=self.L,
            n_dim=self.n_dim,
            pbc=self.pbc,
            max_neighbor_order=self.max_neighbor_order,
        )
        self._built_object = lattice
        return lattice


@dataclass
class TriangularConfig(LatticeConfig):
    type: Literal["Triangular"] = "Triangular"
    L: int = 10
    pbc: bool = True
    max_neighbor_order: int = 2

    def __post_init__(self):
        self.name = f"Triangular_L{self.L}_{'pbc' if self.pbc else ''}"

    def build(self):
        if self._built_object is not None:
            return self._built_object
        assert self.type == "Triangular", "Only Triangular is supported here"
        lattice = nk.graph.Triangular(
            extent=[self.L, self.L],
            pbc=self.pbc,
            max_neighbor_order=self.max_neighbor_order,
        )
        self._built_object = lattice
        return lattice


@dataclass
class HilbertConfig:
    s: float = 0.5
    total_sz: int = None
    _built_object: Optional[object] = field(default=None, init=False, repr=False)

    def build(self, lattice):
        if self._built_object is not None:
            return self._built_object
        hilbert = nk.hilbert.Spin(s=self.s, N=lattice.n_nodes, total_sz=self.total_sz)
        self._built_object = hilbert
        return hilbert


@dataclass
class HamiltonianConfig:
    name: str = "AbstractHamiltonian"
    _built_object: Optional[object] = field(default=None, init=False, repr=False)

    def build(self, hilbert, lattice):
        raise NotImplementedError(f"Hamiltonian type {self.name} not implemented")


@dataclass
class HeisenbergConfig(HamiltonianConfig):
    sign_rule: Tuple[bool, bool] = (False, False)

    def __post_init__(self):
        self.name = "Heisenberg"

    def build(self, hilbert, lattice):
        if self._built_object is not None:
            return self._built_object
        hamiltonian = nk.operator.Heisenberg(
            hilbert=hilbert, graph=lattice, sign_rule=self.sign_rule, J=0.25
        ).to_jax_operator()
        self._built_object = hamiltonian.to_jax_operator()
        return hamiltonian


@dataclass
class J1J2Config(HamiltonianConfig):
    J: Tuple[float, float] = (1.0, 0.5)
    sign_rule: Tuple[bool, bool] = (False, False)

    def __post_init__(self):
        self.name = "J1J2"

    def build(self, hilbert, lattice):
        if self._built_object is not None:
            return self._built_object
        hamiltonian = nk.operator.Heisenberg(
            hilbert=hilbert,
            graph=lattice,
            J=list(self.J),
            sign_rule=self.sign_rule,
        ).to_jax_operator()
        self._built_object = hamiltonian
        return hamiltonian


@dataclass
class TFIMConfig(HamiltonianConfig):
    J: float = 1.0
    h: float = 1.0

    def __post_init__(self):
        self.name = f"TFIM_J_{self.J:1.6f}_h_{self.h:1.6f}"

    def build(self, hilbert, lattice):
        if self._built_object is not None:
            return self._built_object
        hamiltonian = nk.operator.IsingJax(
            hilbert=hilbert, graph=lattice, h=self.h, J=self.J
        )
        self._built_object = hamiltonian
        return hamiltonian


@dataclass
class SamplerConfig:
    name: Literal["MetropolisExchange"] = "MetropolisExchange"
    d_max: int = 2
    sweep_size: Optional[int] = None  # defaults to lattice.n_nodes
    q_blur: float | None = None
    _built_object: Optional[object] = field(default=None, init=False, repr=False)

    def build(self, hilbert, lattice, n_chains_per_rank: int):
        if self._built_object is not None:
            return self._built_object
        assert self.name == "MetropolisExchange"
        sampler = nk.sampler.MetropolisExchange(
            hilbert=hilbert,
            graph=lattice,
            d_max=self.d_max,
            n_chains_per_rank=n_chains_per_rank,
            sweep_size=self.sweep_size or lattice.n_nodes,
        )
        self._built_object = sampler
        return sampler


def _dtype_from_str(s: str):
    s = s.lower()
    if s in ("complex", "c128", "complex128"):
        return complex
    if s in ("c64", "complex64"):
        return jnp.complex64
    return complex


@dataclass
class ModelConfig:
    name: str = "AbstractModel"
    _built_object: Optional[object] = field(default=None, init=False, repr=False)

    def build(self, lattice):
        raise NotImplementedError(f"Model type {self.name} not implemented")


@dataclass
class RBMSymmConfig(ModelConfig):
    alpha: int = 8
    init_stddev: float = 0.01

    def __post_init__(self):
        self.name = f"RBMSymm_alpha_{self.alpha}_std_{self.init_stddev:1.4f}"

    def build(self, lattice):
        if self._built_object is not None:
            return self._built_object
        model = nk.models.RBMSymm(
            symmetries=lattice.point_group(),
            alpha=self.alpha,
            param_dtype=complex,
            kernel_init=normal(stddev=self.init_stddev),
            hidden_bias_init=normal(stddev=self.init_stddev),
            visible_bias_init=normal(stddev=self.init_stddev),
        )
        self._built_object = model
        return model


@dataclass
class GCNNConfig(ModelConfig):
    layers: int = 1
    features: Tuple[int] = (1,)
    init_stddev: float = 0.01

    def __post_init__(self):
        features_str = "_".join(map(str, self.features))
        self.name = f"GCNN_layers_{self.layers}_features_{features_str}_std_{self.init_stddev:1.4f}"

    def build(self, lattice):
        if self._built_object is not None:
            return self._built_object
        model = nk.models.GCNN(
            symmetries=lattice,
            layers=self.layers,
            features=self.features,
            param_dtype=complex,
            complex_output=True,
            kernel_init=normal(stddev=self.init_stddev),
            bias_init=normal(stddev=self.init_stddev),
        )
        self._built_object = model
        return model


@dataclass
class RBMSymmConfig(ModelConfig):
    alpha: int = 8
    init_stddev: float = 0.01

    def __post_init__(self):
        self.name = f"RBMSymm_alpha_{self.alpha}_std_{self.init_stddev:1.4f}"

    def build(self, lattice):
        if self._built_object is not None:
            return self._built_object
        model = nk.models.RBMSymm(
            symmetries=lattice.point_group(),
            alpha=self.alpha,
            param_dtype=complex,
            kernel_init=normal(stddev=self.init_stddev),
            hidden_bias_init=normal(stddev=self.init_stddev),
            visible_bias_init=normal(stddev=self.init_stddev),
        )
        self._built_object = model
        return model



@dataclass
class ViT2DConfig(ModelConfig):
    patch_size: int = 25
    num_layers: int = 8
    d_model: int = 72
    heads: int = 12

    def __post_init__(self):
        self.name = f"ViTConfig_patch_size{self.patch_size}_nl{self.num_layers}_dm{self.d_model}_heads{self.heads}"

    


# =========================
# Optimizer / SR
# =========================
@dataclass
class OptimizerConfig:
    lr: Schedule = field(default_factory=lambda: ConstantSchedule(value=0.01))
    diag_shift: Schedule = field(default_factory=ExponentialDecaySchedule)
    variance_scaled: bool = False

    def __post_init__(self):
        self.name = f"lr_{self.lr.name}/diag_shift_{self.diag_shift.name}"
        if self.variance_scaled:
            self.name = "variance_scaled_" + self.name

    def build_diag_shift(self):
        return _build_schedule(self.diag_shift)

    def build_lr(self):
        return _build_schedule(self.lr)


@dataclass
class SRConfig:
    chunk_size: Optional[int] = None
    momentum: Optional[bool] = False

    def __post_init__(self):
        if not self.chunk_size in [-1, None]:
            self.name = f"chunk_size_{self.chunk_size}"
        else:
            self.name = "chunk_size_None"
        if self.momentum:
            self.name = self.name + f"momentum_{self.momentum}"


@dataclass
class ExperimentConfig:
    seed: int = 0
    number_of_devices: int = jax.device_count()
    world_size: int = jax.device_count() // jax.local_device_count()
    platform: str = jax.default_backend()
    n_samples: int = 8192
    n_steps: int = 1000
    thermalize_steps: int = 10
    root: str = "./data"
    name: str = "test"

    lattice: LatticeConfig = field(default_factory=LatticeConfig)
    hilbert: HilbertConfig = field(default_factory=HilbertConfig)
    hamiltonian: HamiltonianConfig = field(default_factory=J1J2Config)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    model: ModelConfig = field(default_factory=RBMSymmConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    sr: SRConfig = field(default_factory=SRConfig)

    def build_hamiltonian(self):
        return self.hamiltonian.build(
            self.hilbert.build(self.lattice.build()), self.lattice.build()
        )

    def build_sampler(self):
        n_chains_per_rank = self.n_samples // self.number_of_devices
        return self.sampler.build(
            self.hilbert.build(self.lattice.build()),
            self.lattice.build(),
            n_chains_per_rank=n_chains_per_rank,
        )

    def build_model(self):
        return self.model.build(self.lattice.build())

    # ---- Serialization helpers ----
    def to_dict(self) -> Dict[str, Any]:
        flat: Dict[str, Any] = {}

        def _flatten(obj, prefix: str = ""):
            if hasattr(obj, "__dataclass_fields__"):
                for k in obj.__dataclass_fields__:
                    if k.startswith("_"):
                        continue
                    v = getattr(obj, k)
                    key = f"{prefix}.{k}" if prefix else k
                    _flatten(v, key)
                if hasattr(obj, "type"):
                    type_key = f"{prefix}.type" if prefix else "type"
                    flat[type_key] = getattr(obj, "type")
                return

            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k.startswith("_"):
                        continue
                    key = f"{prefix}.{k}" if prefix else str(k)
                    _flatten(v, key)
                return

            # Treat tuples and lists as leaf values (don't recurse)
            if isinstance(obj, (tuple, list)):
                print(obj)
                flat[prefix] = obj
                return

            flat[prefix] = obj

        _flatten(self)
        return flat

    def config_to_yaml(self, sort_keys: bool = False) -> str:
        """Serialize the full experiment config to a YAML string."""
        return yaml.safe_dump(self.to_dict(), sort_keys=sort_keys)

    def save_config_yaml(self) -> str:
        """Save the full experiment config to a .yaml file and return the path."""
        y = self.config_to_yaml()
        path = self.save_path() + "config.yaml"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(y)
        return path

    def save_path(self) -> str:
        base = self.root.rstrip("/")
        blur_str = (
            f"_q_{self.sampler.q_blur:1.2f}" if self.sampler.q_blur is not None else ""
        )
        suffix = f"{self.name}/{self.hamiltonian.name}/{self.lattice.name}/{self.model.name}/{self.sr.name}/{self.optimizer.name}/ns_{self.n_samples}{blur_str}/seed_{self.seed}/"
        return os.path.join(base, suffix)

    def override(self, updates: Dict[str, Any]) -> "ExperimentConfig":
        cfg = self
        for k, v in updates.items():
            if "." not in k:
                cfg = replace(cfg, **{k: v})
                continue
            head, tail = k.split(".", 1)
            sub = getattr(cfg, head)
            setattr(sub, tail, v) if hasattr(sub, tail) else None
        return cfg

    def __str__(self) -> str:
        return pretty_config(self, 0)


def pretty_config(obj, indent: int = 0) -> str:
    """Human-friendly, deterministic tree rendering of (nested) dataclasses/lists.
    Skips private fields (starting with '_').
    """
    sp = "  " * indent
    if is_dataclass(obj):
        cls = obj.__class__.__name__
        lines = [f"{sp}{cls}:"]
        for name, fielddef in obj.__dataclass_fields__.items():
            if name.startswith("_"):
                continue
            val = getattr(obj, name)
            # compact for scalars and simple tuples
            if is_dataclass(val):
                lines.append(f"{sp}  [{name.upper()}]")
                lines.append(pretty_config(val, indent + 2))
            elif isinstance(val, (list, tuple)):
                lines.append(f"{sp}  {name} [len={len(val)}]:")
                for i, v in enumerate(val):
                    if is_dataclass(v):
                        lines.append(f"{sp}    - ({v.__class__.__name__})")
                        lines.append(pretty_config(v, indent + 3))
                    else:
                        lines.append(f"{sp}    - {v}")
            else:
                lines.append(f"{sp}  {name}: {_fmt_scalar(val)}")
        return "\n".join(lines)
    elif isinstance(obj, dict):
        lines = [f"{sp}dict:"]
        for k in sorted(obj):
            lines.append(f"{sp}  {k}: {pretty_config(obj[k], indent + 2).lstrip()}")
        return "\n".join(lines)
    elif isinstance(obj, (list, tuple)):
        lines = [f"{sp}list[len={len(obj)}]:"]
        for v in obj:
            lines.append(pretty_config(v, indent + 1))
        return "\n".join(lines)
    else:
        return f"{sp}{_fmt_scalar(obj)}"


def _fmt_scalar(x):
    if isinstance(x, float):
        return f"{x:.6g}"
    if isinstance(x, complex):
        r = f"{x.real:.6g}"
        i = f"{x.imag:.6g}"
        return f"({r}+{i}j)"
    return str(x)
