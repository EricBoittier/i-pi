"""Compilation of the MD steps with jax.jit (`--jit`)."""

# This file is part of i-PI.
# i-PI Copyright (C) 2014-2026 i-PI developers
# See the "licenses" directory for full license information.

import numpy as np
import pytest

import ipi.utils.array_backend as ab
from ipi_tests.unit_tests.engine.test_backend_equivalence import (
    HARMONIC,
    NO_FORCES,
    write_input,
)

pytest.importorskip("jax")

DUMMY = ("<ffdirect name='ff'><pes>dummy</pes></ffdirect>", "<force forcefield='ff'/>")
PROPERTIES = ["kinetic_md", "potential", "conserved", "temperature"]
# the compiled steps may sum in a different order than the eager ones
TOLERANCE = dict(rtol=1e-12, atol=1e-14)


@pytest.fixture
def numpy_afterwards():
    """Restores the numpy backend after the test."""

    yield
    ab.set_backend("numpy")


def simulation(xml, jit):
    from ipi.scripting import InteractiveSimulation

    return InteractiveSimulation(xml, backend="jax", jit=jit)


def jit_step(sim):
    """The object compiling the steps of the first system, if any."""

    from ipi.engine.jit import JitStep

    step = getattr(getattr(sim.syslist[0].motion, "integrator", None), "step", None)
    return step if isinstance(step, JitStep) else None


def assert_same_state(a, b):
    for name in PROPERTIES:
        np.testing.assert_allclose(b.properties(name), a.properties(name), **TOLERANCE)
    a, b = a.syslist[0], b.syslist[0]
    for x, y in [(a.beads.q, b.beads.q), (a.beads.p, b.beads.p), (a.cell.h, b.cell.h)]:
        np.testing.assert_allclose(ab.to_numpy(y), ab.to_numpy(x), **TOLERANCE)


@pytest.mark.parametrize(
    "case,forcefield",
    [
        ("nve", DUMMY),
        ("nvt_langevin", NO_FORCES),
        ("nvt_svr", NO_FORCES),
        ("pimd_nvt", DUMMY),
        ("pimd_pile_g", DUMMY),
        ("npt_isotropic", DUMMY),
        ("npt_flexible", NO_FORCES),
    ],
)
def test_compiled_steps_match_eager_steps(
    case, forcefield, tmp_path, monkeypatch, numpy_afterwards
):
    """The compiled steps give the same trajectory and outputs as eager steps
    (to rounding errors), and draw the same random numbers"""

    monkeypatch.chdir(tmp_path)
    xml = write_input(case, forcefield)
    eager, compiled = simulation(xml, False), simulation(xml, True)
    for step in range(8):
        eager.run_step(step)
        compiled.run_step(step)
        assert_same_state(eager, compiled)
    assert jit_step(compiled).compiled is not None
    assert compiled.prng.get_state()["jax"] == eager.prng.get_state()["jax"]


def test_forces_of_numpy_drivers_are_eager(tmp_path, monkeypatch, numpy_afterwards):
    """Steps that need the forces of a driver working on numpy arrays are not
    compiled, and run eagerly"""

    monkeypatch.chdir(tmp_path)
    xml = write_input("nvt_langevin", HARMONIC)
    eager, compiled = simulation(xml, False), simulation(xml, True)
    for step in range(5):
        eager.run_step(step)
        compiled.run_step(step)
    assert jit_step(compiled).compiled is None
    assert "forcefield 'ff'" in jit_step(compiled).reason
    assert_same_state(eager, compiled)


def test_temperature_changes_reach_the_compiled_step(
    tmp_path, monkeypatch, numpy_afterwards
):
    """The coefficients of the thermostat are arrays, recomputed when the
    temperature changes and passed to the compiled step"""

    monkeypatch.chdir(tmp_path)
    xml = write_input("pimd_nvt", NO_FORCES)
    eager, compiled = simulation(xml, False), simulation(xml, True)
    for step in range(8):
        if step == 5:
            for sim in (eager, compiled):
                sim.syslist[0].ensemble.temp *= 2.0
        eager.run_step(step)
        compiled.run_step(step)
    assert_same_state(eager, compiled)


@pytest.mark.parametrize(
    "case,owner,name",
    [("nvt_langevin", "motion", "dt"), ("npt_isotropic", "ensemble", "pext")],
)
def test_changed_parameters_stop_the_compiled_steps(
    case, owner, name, tmp_path, monkeypatch, numpy_afterwards
):
    """The time step, and the pressure of the barostats, are fixed in the
    compiled step"""

    from ipi.utils.jit import JitError

    monkeypatch.chdir(tmp_path)
    sim = simulation(write_input(case, NO_FORCES), True)
    for step in range(4):
        sim.run_step(step)
    obj = getattr(sim.syslist[0], owner)
    setattr(obj, name, getattr(obj, name) * 1.1)
    with pytest.raises(JitError, match="cannot change"):
        sim.run_step(4)


# a harmonic driver on JAX arrays that cannot be traced: it reads its inputs
# as numpy arrays, which fails on the tracers of jax.jit
UNTRACEABLE = """
import numpy as np
import jax.numpy as jnp
from ipi.pes.dummy import Dummy_driver

__DRIVER_NAME__ = "untraceable"
__DRIVER_CLASS__ = "Untraceable"
CALLS = []


class Untraceable(Dummy_driver):
    backend_arrays = True
    traceable = False

    def check_parameters(self):
        # device constants: Python scalars would be copied to the device
        self.k, self.zero = jnp.asarray(1e-3), jnp.zeros((3, 3))

    def compute_structure(self, cell, pos):
        CALLS.append(np.asarray(pos).copy())
        kx = self.k * pos
        return jnp.vdot(kx, pos) / (self.k / self.k + self.k / self.k), -kx, self.zero, ""
"""


@pytest.mark.parametrize("case", ["nve", "nvt_svr", "npt_isotropic"])
def test_untraceable_drivers_run_eagerly_in_compiled_steps(
    case, tmp_path, monkeypatch, numpy_afterwards
):
    """A driver that works on JAX arrays but cannot be traced is called eagerly
    at each compiled step, on the current positions, without host transfers,
    and gives the trajectory of the eager steps"""

    import sys

    import jax

    monkeypatch.chdir(tmp_path)
    (tmp_path / "untraceable.py").write_text(UNTRACEABLE)
    forcefield = (
        "<ffdirect name='ff'><pes>custom</pes>"
        f"<pes_path>{tmp_path / 'untraceable.py'}</pes_path></ffdirect>",
        "<force forcefield='ff'/>",
    )
    xml = write_input(case, forcefield)
    eager, compiled = simulation(xml, False), simulation(xml, True)
    # the module of the driver of the compiled simulation, loaded last
    calls = sys.modules["untraceable"].CALLS
    for step in range(4):
        eager.run_step(step)
        compiled.run_step(step)
    assert jit_step(compiled).compiled is not None
    before = len(calls)
    for step in range(4, 8):
        eager.run_step(step)
        with jax.transfer_guard("disallow"):
            compiled.run_step(step)
    assert_same_state(eager, compiled)
    # one eager call per compiled step, on the positions of that step
    assert len(calls) == before + 4
    np.testing.assert_allclose(
        calls[-1].reshape(-1), ab.to_numpy(compiled.syslist[0].beads.q).reshape(-1)
    )
