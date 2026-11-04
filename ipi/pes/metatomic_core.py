"""
Interface with metatomic-core models (https://docs.metatensor.org/metatomic/), i.e.
any model implementing the metatomic C API: TorchScript models loaded through the
``metatomic-torch`` plugin, JAX/NumPy models defined in Python, or native plugins.

Unlike the ``metatomic`` driver (TorchScript only, through ``metatomic.torch``),
this driver only depends on the backend-independent ``metatomic`` (metatomic-core)
Python package, ``metatensor`` and ``vesin``.
"""

import importlib.util
import pathlib
import sys

import numpy as np

from ipi.utils.messages import info, warning
from ipi.utils.units import unit_to_internal, unit_to_user
from .dummy import Dummy_driver

mta = None
metatensor = None
vesin = None

__DRIVER_NAME__ = "metatomic_core"
__DRIVER_CLASS__ = "MetatomicCoreDriver"


def _as_bool(value):
    return value if isinstance(value, bool) else str(value).lower() in ("1", "true")


class MetatomicCoreDriver(Dummy_driver):
    """
    Driver for models implementing the metatomic C API (metatomic-core).

    Command-line:
        i-pi-py_driver -m metatomic_core -o template=template.xyz,model=<model> [...]

    Parameters:
        :param template: string, ASE-readable structure file giving the atomic types
            and the periodic boundary conditions
        :param model: string, the model to run. Either ``path/to/file.py:name``, where
            ``name`` is a ``metatomic.BaseModel`` subclass (instantiated without
            arguments), a function returning a model, or a model instance; or any
            string accepted by ``metatomic.load_model`` (e.g. a ``.pt`` file for the
            metatomic-torch plugin, or the model name of a native plugin)
        :param plugins: string, optional, ``;``-separated list of plugin shared
            libraries to load before loading the model. Bare names are looked up in
            metatomic's ``libexec/metatomic`` directory, and ``torch`` stands for the
            metatomic-torch plugin. The torch plugin is loaded automatically for
            ``.pt`` models.
        :param plugin_name: string, optional, the plugin ``metatomic.load_model``
            should use
        :param option_<key>: optional, string option ``key`` given to
            ``metatomic.load_model`` (e.g. ``option_cutoff=7.0``)
        :param energy_variant: string, optional, which ``energy/<variant>`` output
            to use for a multi-head model
        :param check_consistency: bool, optional, whether metatomic should check the
            data given to and returned by the model
        :param enable_x64: bool, optional (default true), enable 64-bit floats in JAX
            before the model is constructed (JAX models cast their parameters to
            ``float64`` on construction)
        :param skin: float, optional, neighbor list skin in Angstrom (default 2.0)
        :param array: str, optional, array type given to the model: ``numpy``
            (default, host arrays), ``torch`` (torch tensors on ``device``) or ``jax``
            (JAX arrays on JAX's default device). Device arrays avoid the host
            round-trip for models that run on a GPU (TorchScript, JAX); native
            plugins copy host arrays to their own device.
        :param device: str, optional, torch device for ``array=torch`` (default cuda)

    With ``array=backend``, the driver works directly on the arrays of the i-PI
    array backend (``i-pi --backend jax``, in-process ``ffdirect`` only), which
    stay on their device: the positions are given to the model, and the forces
    returned to i-PI, without copies or host transfers. The pairs come from a
    Verlet list (cutoff + ``skin``) whose indices are kept on the device, and
    which is rebuilt (on the host, with vesin) when an atom has moved by more
    than half the skin, or the cell changed by as much. This is checked every
    ``nl_check`` calls, reading one number from the device: the other calls make
    no host transfer. With ``--jit``, the driver is called eagerly within the
    compiled steps, unless ``trace=true``: a JAX model is then traced in the
    compiled step (through ``JaxModel.evaluate``) with the pair list of the last
    eager call, which must remain valid (e.g. a molecule within the cutoff).
    Pair lists that the model requests as ``strict`` get no skin, i.e. are
    rebuilt at each call with exactly the pairs within the cutoff.
    A JAX model that provides ``jax_pair_arrays`` and ``jax_energy_function``
    (padded pair arrays as data, e.g. the pet-jax adapter) is called through a
    compiled function whose pair arrays are padded to a fixed capacity
    (``pair_capacity_factor`` times the first list, 0 to disable): rebuilding
    the list then re-uses the compiled function, which is only re-compiled when
    the capacity overflows (growing by 1.5). With ``device_rebuild=true`` (and a
    model that provides ``jax_device_pairs``), the half-skin check and the
    rebuild run on the device too, in the same compiled call, so that only the
    overflow flag is read, every ``nl_check`` calls (brute force over the
    periodic images: small systems).
    """

    def __init__(
        self,
        template,
        model,
        plugins=None,
        plugin_name=None,
        energy_variant=None,
        check_consistency=False,
        enable_x64=True,
        skin=2.0,
        array="numpy",
        device="cuda",
        nl_check=1,
        trace=False,
        pair_capacity_factor=1.3,
        device_rebuild=False,
        *args,
        **kwargs,
    ):
        global mta, metatensor, vesin
        try:
            import metatomic as mta
            import metatensor
            import vesin
        except ImportError as e:
            message = (
                "could not find the metatomic_core driver dependencies, make sure "
                "they are installed with `python -m pip install metatomic vesin ase`"
            )
            warning(f"{message}: {e}")
            raise ImportError(message) from e

        self.template = template
        self.model_spec = str(model)
        self.plugins = [p for p in str(plugins or "").split(";") if p.strip()]
        self.plugin_name = plugin_name
        self.model_options = {
            key[len("option_") :]: str(kwargs.pop(key))
            for key in list(kwargs)
            if key.startswith("option_")
        }
        self.energy_variant = energy_variant
        self.check_consistency = _as_bool(check_consistency)
        self.enable_x64 = _as_bool(enable_x64)
        self.skin = float(skin)
        self.array = str(array)
        self.device = str(device)
        if self.array not in ("numpy", "torch", "jax", "backend"):
            raise ValueError(
                f"array must be numpy, torch, jax or backend, got {self.array}"
            )
        self.backend_arrays = self.array == "backend"
        self.traceable = _as_bool(trace)
        self.nl_check = max(int(nl_check), 1)
        self.pair_capacity_factor = float(pair_capacity_factor)
        self.device_rebuild = _as_bool(device_rebuild)
        super().__init__(*args, **kwargs)

    def check_parameters(self):
        import ase.io

        atoms = ase.io.read(self.template)
        self._types = atoms.numbers.astype(np.int32)
        self._pbc = np.array(atoms.pbc, dtype=bool)

        # outputs of native plugins use the C++ metatensor array type, older
        # metatensor releases need to be told how to read them from Python
        for origin in (
            "metatensor::SimpleDataArray",
            "metatensor_torch::TorchDataArray",
        ):
            try:
                metatensor.register_external_data_wrapper(
                    origin, metatensor.ExternalCpuArray
                )
            except Exception:
                pass

        self.model = self._load_model()
        capabilities = self.model.capabilities()
        self._dtype = np.dtype(capabilities.dtype)
        self._energy = self._energy_request(capabilities)
        self._pair_lists = self.model.requested_pair_lists([self._energy])
        inputs = self.model.requested_inputs([self._energy])
        if len(inputs) != 0:
            raise ValueError(
                "the metatomic_core driver can not provide the additional inputs "
                f"requested by this model: {[q.name for q in inputs]}"
            )
        length = mta.unit_conversion_factor(capabilities.length_unit, "angstrom")
        self._neighbors = [
            vesin.NeighborList(
                cutoff=options.cutoff * length,
                full_list=options.full_list,
                skin=self.skin,
            )
            for options in self._pair_lists
        ]
        unknown = set(self._types.tolist()) - set(capabilities.atomic_types)
        if unknown:
            raise ValueError(f"the model does not support atomic types {unknown}")
        info(f"metatomic-core model:\n{self.model.metadata()}", self.verbose)
        info(f"computing {self._energy}", self.verbose)

    def _load_model(self):
        if self.enable_x64:
            try:  # must happen before any JAX array is created by the model
                import jax

                jax.config.update("jax_enable_x64", True)
            except ImportError:
                pass

        path, _, name = self.model_spec.rpartition(":")
        if path.endswith(".py"):
            path = pathlib.Path(path).resolve()
            sys.path.insert(0, str(path.parent))
            spec = importlib.util.spec_from_file_location(path.stem, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[path.stem] = module
            spec.loader.exec_module(module)
            model = getattr(module, name)
            if not isinstance(model, mta.BaseModel):
                model = model()
            if not isinstance(model, mta.BaseModel):
                raise TypeError(f"{self.model_spec} is not a metatomic.BaseModel")
            return model

        plugins = list(self.plugins)
        if self.model_spec.endswith(".pt") and "torch" not in plugins:
            plugins.append("torch")
        for plugin in plugins:
            mta.load_plugin(self._plugin_path(plugin.strip()))
        return mta.load_model(
            self.model_spec,
            options=self.model_options or None,
            plugin_name=self.plugin_name,
        )

    @staticmethod
    def _plugin_path(plugin):
        if plugin == "torch":
            import metatomic_torch

            return metatomic_torch.utils.plugin_path
        if pathlib.Path(plugin).exists():
            return plugin
        libexec = pathlib.Path(mta.__file__).parent / "libexec" / "metatomic"
        candidates = [libexec / plugin, libexec / f"{plugin}.so"]
        candidates += [libexec / f"metatomic-{plugin}-plugin.so"]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        raise ValueError(f"could not find metatomic plugin '{plugin}'")

    def _energy_request(self, capabilities):
        name = (
            "energy" if self.energy_variant is None else f"energy/{self.energy_variant}"
        )
        candidates = [q for q in capabilities.outputs if q.name == name]
        if not candidates and self.energy_variant is None:
            # a model with a single, variant-only energy output
            candidates = [
                q for q in capabilities.outputs if q.name.startswith("energy/")
            ]
            if len({q.name for q in candidates}) > 1:
                raise ValueError(
                    "the model has several energy variants, select one with "
                    f"energy_variant: {sorted({q.name for q in candidates})}"
                )
        if not candidates:
            raise ValueError(f"the model does not provide a '{name}' output")
        # prefer per-system energies, summed per-atom energies otherwise
        declared = sorted(candidates, key=lambda q: q.sample_kind != "system")[0]
        if "positions" not in declared.gradients:
            raise ValueError(f"'{declared.name}' has no gradients w.r.t. positions")
        gradients = [g for g in ("positions", "strain") if g in declared.gradients]
        if "strain" not in gradients:
            warning("the model has no strain gradients: the virial will be zero")
        return mta.Quantity(
            name=declared.name,
            unit="eV",
            sample_kind=declared.sample_kind,
            gradients=gradients,
        )

    def _arr(self, x, dtype=None):
        x = np.ascontiguousarray(x, dtype=dtype)
        if self.array == "torch":
            import torch

            return torch.as_tensor(x, device=self.device)
        if self.array == "jax":
            import jax.numpy as jnp

            return jnp.asarray(x)
        return x

    def _labels(self, names, values):
        # torch blocks need their Labels on the same device as the values
        if self.array == "torch":
            import torch

            values = torch.as_tensor(values, device=self.device)
        return metatensor.Labels(names, values)

    def _system(self, cell, pos):
        positions = unit_to_user("length", "angstrom", pos).reshape(-1, 3)
        # i-PI stores cell vectors as columns, metatomic as rows
        cell = unit_to_user("length", "angstrom", cell.T).copy()
        cell[~self._pbc] = 0.0
        if len(positions) != len(self._types):
            raise ValueError(
                f"got {len(positions)} atoms, the template has {len(self._types)}"
            )
        system = mta.System(
            "angstrom",
            self._arr(self._types),
            self._arr(positions, self._dtype),
            self._arr(cell, self._dtype),
            self._arr(self._pbc),
        )
        for options, neighbors in zip(self._pair_lists, self._neighbors):
            i, j, shifts, vectors = neighbors.compute(
                points=positions,
                box=cell,
                periodic=self._pbc,
                quantities="ijSD",
                copy=True,
            )
            samples = self._labels(
                [
                    "first_atom",
                    "second_atom",
                    "cell_shift_a",
                    "cell_shift_b",
                    "cell_shift_c",
                ],
                np.column_stack([i, j, shifts]).astype(np.int32).reshape(-1, 5),
            )
            pairs = metatensor.TensorBlock(
                self._arr(vectors, self._dtype).reshape(-1, 3, 1),
                samples,
                [self._labels(["xyz"], np.arange(3, dtype=np.int32)[:, None])],
                self._labels(["distance"], np.zeros((1, 1), dtype=np.int32)),
            )
            system.add_pairs(options, pairs)
        return system

    def compute(self, cell, pos):
        """Evaluates one structure, or a list of structures in a single model call"""
        if self.backend_arrays and not isinstance(
            pos if not isinstance(pos, list) else pos[0], np.ndarray
        ):
            if isinstance(cell, list):
                return [self._native(c, p) for c, p in zip(cell, pos)]
            return self._native(cell, pos)
        batched = isinstance(cell, list)
        cells, positions = (cell, pos) if batched else ([cell], [pos])
        if len(cells) != len(positions):
            raise ValueError("cell and pos lists should have the same length")

        systems = [self._system(c, p) for c, p in zip(cells, positions)]
        (energy,) = mta.execute_model(
            self.model,
            systems,
            [self._energy],
            check_consistency=self.check_consistency,
        )
        block = energy.block().to(arrays="numpy")
        n_systems = len(systems)

        energies = np.zeros(n_systems)
        np.add.at(energies, block.samples.column("system"), block.values[:, 0])

        gradient = block.gradient("positions")
        forces = np.zeros((n_systems, len(self._types), 3))
        np.add.at(
            forces,
            (gradient.samples.column("system"), gradient.samples.column("atom")),
            -np.asarray(gradient.values)[..., 0],
        )

        virials = np.zeros((n_systems, 3, 3))
        if "strain" in block.gradients_list():
            gradient = block.gradient("strain")
            parent = block.samples.column("system")[gradient.samples.column("sample")]
            np.add.at(virials, parent, -np.asarray(gradient.values)[..., 0])
            # models that are not exactly rotationally equivariant (e.g. PET) give a
            # slightly non-symmetric virial: keep its symmetric part, as the
            # metatomic driver does
            virials = (virials + virials.swapaxes(1, 2)) * 0.5

        results = [
            (
                unit_to_internal("energy", "electronvolt", e),
                unit_to_internal("force", "ev/ang", f).reshape(np.shape(p)),
                unit_to_internal("energy", "electronvolt", v.T),
                "",
            )
            for e, f, v, p in zip(energies, forces, virials, positions)
        ]
        return results if batched else results[0]

    def compute_structure(self, cell, pos):
        return self.compute(cell, pos)

    def _native(self, cell, pos):
        """One structure, as JAX arrays of the i-PI backend (see ``array=backend``)"""
        import jax
        import jax.numpy as jnp

        if not hasattr(self, "_jx"):
            constant = lambda value: jnp.asarray(value)  # noqa: E731
            self._jx = dict(
                length=constant(unit_to_user("length", "angstrom", 1.0)),
                energy=constant(unit_to_internal("energy", "electronvolt", 1.0)),
                force=constant(unit_to_internal("force", "ev/ang", 1.0)),
                types=jnp.asarray(self._types),
                pbc=jnp.asarray(self._pbc),
                zero=jnp.zeros(()),
                half=constant(0.5),
                half_skin=constant(self.skin / 2),
                lists=[None] * len(self._pair_lists),
                calls=0,
                padded=self.pair_capacity_factor > 0
                and len(self._pair_lists) == 1
                and hasattr(self.model, "jax_energy_function"),
                fn=None,
                capacity=None,
                overflow=None,
                compiles=0,
                rebuilds=0,
                regrowths=0,
            )
        jx = self._jx
        x = pos.reshape(-1, 3) * jx["length"]
        c = jnp.where(jx["pbc"][:, None], cell.T * jx["length"], jx["zero"])
        traced = isinstance(x, jax.core.Tracer)
        if jx["lists"][0] is not None and self._pbc.any():
            # undo the folding of the positions into the cell since the list was
            # built (e.g. ffdirect pbc=true), which would invalidate its shifts
            device = jx.get("device")
            x_ref = jx["lists"][0]["x"] if device is None else device["state"]["x"]
            x = x - jnp.round((x - x_ref) @ jnp.linalg.inv(c)) @ c
        if not traced:
            jx["calls"] += 1
            check = jx["calls"] % self.nl_check == 0
            if jx.get("device") is not None:
                # rebuilds run on the device: only the overflow flag is read here
                if check and bool(jx["overflow"]):
                    jx["device"] = None
                    jx["lists"][0] = self._verlet(0, x, c)
                    jx["rebuilds"] += 1
                    self._pad(jx["lists"][0])
                check = False
            for k, nl in enumerate(jx["lists"]):
                if jx.get("device") is not None:
                    break
                if nl is None or nl["skin"] == 0 or (check and self._moved(nl, x, c)):
                    jx["lists"][k] = self._verlet(k, x, c)
                    jx["rebuilds"] += 1
                    if jx["padded"]:
                        self._pad(jx["lists"][k])
        if jx["padded"] and self.device_rebuild and jx["fn"] is not None:
            if jx.get("device") is None:
                self._device_step(jx["lists"][0])
            out = jx["device"]["call"](x, c, jx["device"]["state"])
            e, f, v, overflow, jx["device"]["state"] = out
            jx["overflow"] = overflow
            v = (v + v.T) * jx["half"]
            return (
                e * jx["energy"],
                (f * jx["force"]).reshape(pos.shape),
                v.T * jx["energy"],
                "",
            )
        if jx["padded"]:
            e, f, v, jx["overflow"] = jx["fn"](x, c, **jx["lists"][0]["pairs"])
            v = (v + v.T) * jx["half"]
            return (
                e * jx["energy"],
                (f * jx["force"]).reshape(pos.shape),
                v.T * jx["energy"],
                "",
            )
        pairs = [
            (options, nl["samples"], x[nl["j"]] - x[nl["i"]] + nl["S"] @ c)
            for options, nl in zip(self._pair_lists, jx["lists"])
        ]
        if traced:
            system = mta.JaxSystem(
                self._types,
                x,
                c,
                self._pbc,
                pairs={o: (s, v.reshape(-1, 3, 1)) for o, s, v in pairs},
            )
            (energy,) = self.model.evaluate([system], [self._energy])
        else:
            system = mta.System("angstrom", jx["types"], x, c, jx["pbc"])
            for options, samples, vectors in pairs:
                system.add_pairs(
                    options,
                    metatensor.TensorBlock(
                        vectors.reshape(-1, 3, 1),
                        samples,
                        [metatensor.Labels.range("xyz", 3)],
                        metatensor.Labels.range("distance", 1),
                    ),
                )
            (energy,) = mta.execute_model(
                self.model,
                [system],
                [self._energy],
                check_consistency=self.check_consistency,
            )
        block = energy.block()
        gradient = block.gradient("positions")
        atoms = np.asarray(gradient.samples.column("atom"))
        forces = -gradient.values[..., 0]
        if not np.array_equal(atoms, np.arange(len(self._types))):
            forces = (
                jnp.zeros((len(self._types), 3), forces.dtype).at[atoms].add(forces)
            )
        virial = jnp.zeros((3, 3), forces.dtype)
        if "strain" in block.gradients_list():
            virial = -block.gradient("strain").values[..., 0].sum(axis=0)
            virial = (virial + virial.T) * jx["half"]
        return (
            block.values.sum() * jx["energy"],
            (forces * jx["force"]).reshape(pos.shape),
            virial.T * jx["energy"],
            "",
        )

    def _verlet(self, k, x, c):
        """Builds the Verlet list `k` (cutoff + skin) on the host, keeping its
        indices on the device"""
        import jax.numpy as jnp

        options = self._pair_lists[k]
        length = mta.unit_conversion_factor(
            self.model.capabilities().length_unit, "angstrom"
        )
        # a strict pair list must hold exactly the pairs within the cutoff
        skin = 0.0 if options.strict else self.skin
        i, j, S = vesin.NeighborList(
            cutoff=options.cutoff * length + skin, full_list=options.full_list
        ).compute(
            points=np.asarray(x),
            box=np.asarray(c),
            periodic=self._pbc,
            quantities="ijS",
            copy=True,
        )
        samples = metatensor.Labels(
            [
                "first_atom",
                "second_atom",
                "cell_shift_a",
                "cell_shift_b",
                "cell_shift_c",
            ],
            np.column_stack([i, j, S]).astype(np.int32).reshape(-1, 5),
        )
        nl = dict(skin=skin, host=(i, j, S), samples=samples, x=x, c=c)
        if not self._jx["padded"]:  # the padded path uses the model's own arrays
            nl.update(
                i=jnp.asarray(i.astype(np.int32)),
                j=jnp.asarray(j.astype(np.int32)),
                S=jnp.asarray(S.astype(x.dtype)),
            )
        return nl

    def _moved(self, nl, x, c):
        """Whether an atom moved, or the cell changed, by more than half the skin
        since the list was built: one number read from the device"""
        import jax.numpy as jnp

        moved = jnp.max(jnp.abs(x - nl["x"])) + jnp.max(jnp.abs(c - nl["c"]))
        moved = moved > self._jx["half_skin"]
        if self._jx["overflow"] is not None:  # PET's per-atom edge capacity
            moved = moved | self._jx["overflow"]
        return bool(moved)

    def _device_step(self, nl):
        """One compiled call with the half-skin check and the rebuild on the device:
        (positions, cell, state) -> (energy, forces, virial, overflow, state), the
        state holding the pair arrays, the reference positions and cell, and the
        number of rebuilds"""
        import jax
        import jax.numpy as jnp

        jx = self._jx
        options = self._pair_lists[0]
        length = mta.unit_conversion_factor(
            self.model.capabilities().length_unit, "angstrom"
        )
        build = self.model.jax_device_pairs(
            len(self._types),
            jx["capacity"][0],
            options.cutoff * length + self.skin,
            bool(self._pbc.any()),
        )
        energy, half_skin = jx["fn"], jx["half_skin"]

        @jax.jit
        def call(x, c, state):
            moved = jnp.max(jnp.abs(x - state["x"])) + jnp.max(jnp.abs(c - state["c"]))

            def rebuild(_):
                arrays, overflow = build(x, c)
                return dict(
                    arrays=arrays,
                    x=x,
                    c=c,
                    overflow=overflow,
                    rebuilds=state["rebuilds"] + 1,
                )

            state = jax.lax.cond(moved > half_skin, rebuild, lambda s: s, state)
            e, f, v, overflow = energy(x, c, **state["arrays"])
            return e, f, v, overflow | state["overflow"], state

        state = dict(
            arrays=nl["pairs"],
            x=nl["x"],
            c=nl["c"],
            overflow=jnp.zeros((), bool),
            rebuilds=jnp.zeros((), jnp.int32),
        )
        jx["device"] = dict(call=call, state=state)

    def _pad(self, nl):
        """The pair arrays of the Verlet list `nl` padded to the capacity of the
        compiled energy function, which is (re-)compiled when it overflows"""
        import math

        import jax.numpy as jnp

        jx = self._jx
        i, j, S = nl["host"]
        n = len(self._types)
        pbc = bool(self._pbc.any())
        count = 0
        if jx["capacity"] is None:
            # the per-atom edge count after PET's selection sizes the first capacity;
            # later lists rely on the model's own overflow flag (computing it again
            # would compile the selection for each new pair count)
            x, c = np.asarray(nl["x"]), np.asarray(nl["c"])
            _, count = self.model.jax_pair_arrays(i, j, S, n, len(i) + 1, x, c, pbc)
        if jx["capacity"] is None:
            factor = self.pair_capacity_factor
            jx["capacity"] = [math.ceil(factor * len(i)) + 1, math.ceil(factor * count)]
        overflow = jx["overflow"] is not None and bool(jx["overflow"])
        while len(i) >= jx["capacity"][0] or count > jx["capacity"][1] or overflow:
            jx["capacity"] = [math.ceil(1.5 * c) for c in jx["capacity"]]
            jx["fn"], jx["regrowths"], overflow = None, jx["regrowths"] + 1, False
            info(
                f"metatomic_core: pair capacity grown to {jx['capacity']}", self.verbose
            )
        if jx["fn"] is None:
            jx["fn"] = self.model.jax_energy_function(
                self._types, pbc, jx["capacity"][1]
            )
            jx["compiles"] += 1
        arrays, _ = self.model.jax_pair_arrays(i, j, S, n, jx["capacity"][0])
        nl["count"] = count
        nl["pairs"] = {key: jnp.asarray(value) for key, value in arrays.items()}
