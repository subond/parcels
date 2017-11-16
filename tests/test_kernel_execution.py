from parcels import (
    FieldSet, ParticleSet, ScipyParticle, JITParticle, ErrorCode, KernelError,
    OutOfBoundsError
)
import numpy as np
import pytest


ptype = {'scipy': ScipyParticle, 'jit': JITParticle}


def DoNothing(particle, fieldset, time, dt):
    return ErrorCode.Success


@pytest.fixture
def fieldset(xdim=20, ydim=20):
    """ Standard unit mesh fieldset """
    lon = np.linspace(0., 1., xdim, dtype=np.float32)
    lat = np.linspace(0., 1., ydim, dtype=np.float32)
    U, V = np.meshgrid(lat, lon)
    data = {'U': np.array(U, dtype=np.float32), 'V': np.array(V, dtype=np.float32)}
    dimensions = {'lat': lat, 'lon': lon}
    return FieldSet.from_data(data, dimensions, mesh='flat')


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('start, end, substeps, dt', [
    (0., 10., 1, 1.),
    (0., 10., 4, 1.),
    (0., 10., 1, 3.),
    (2., 16., 5, 3.),
    (20., 10., 4, -1.),
    (20., -10., 7, -2.),
])
def test_execution_endtime(fieldset, mode, start, end, substeps, dt, npart=10):
    pset = ParticleSet(fieldset, pclass=ptype[mode],
                       lon=np.linspace(0, 1, npart, dtype=np.float32),
                       lat=np.linspace(1, 0, npart, dtype=np.float32))
    pset.execute(DoNothing, starttime=start, endtime=end, dt=dt)
    assert np.allclose(np.array([p.time for p in pset]), end)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('start, end, substeps, dt', [
    (0., 10., 1, 1.),
    (0., 10., 4, 1.),
    (0., 10., 1, 3.),
    (2., 16., 5, 3.),
    (20., 10., 4, -1.),
    (20., -10., 7, -2.),
])
def test_execution_runtime(fieldset, mode, start, end, substeps, dt, npart=10):
    pset = ParticleSet(fieldset, pclass=ptype[mode],
                       lon=np.linspace(0, 1, npart, dtype=np.float32),
                       lat=np.linspace(1, 0, npart, dtype=np.float32))
    t_step = (end - start) / substeps
    for _ in range(substeps):
        pset.execute(DoNothing, starttime=start, runtime=t_step, dt=dt)
        start += t_step
    assert np.allclose(np.array([p.time for p in pset]), end)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_execution_fail_timed(fieldset, mode, npart=10):
    def TimedFail(particle, fieldset, time, dt):
        if particle.time >= 10.:
            return ErrorCode.Error
        else:
            return ErrorCode.Success

    pset = ParticleSet(fieldset, pclass=ptype[mode],
                       lon=np.linspace(0, 1, npart, dtype=np.float32),
                       lat=np.linspace(1, 0, npart, dtype=np.float32))
    error_thrown = False
    try:
        pset.execute(TimedFail, starttime=0., endtime=20., dt=2.)
    except KernelError:
        error_thrown = True
    assert error_thrown
    assert len(pset) == npart
    assert np.allclose(np.array([p.time for p in pset]), 10.)


@pytest.mark.parametrize('mode', ['scipy'])
def test_execution_fail_python_exception(fieldset, mode, npart=10):
    def PythonFail(particle, fieldset, time, dt):
        if particle.time >= 10.:
            raise RuntimeError("Enough is enough!")
        else:
            return ErrorCode.Success

    pset = ParticleSet(fieldset, pclass=ptype[mode],
                       lon=np.linspace(0, 1, npart, dtype=np.float32),
                       lat=np.linspace(1, 0, npart, dtype=np.float32))
    error_thrown = False
    try:
        pset.execute(PythonFail, starttime=0., endtime=20., dt=2.)
    except KernelError:
        error_thrown = True
    assert error_thrown
    assert len(pset) == npart
    assert np.allclose(np.array([p.time for p in pset]), 10.)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_execution_fail_out_of_bounds(fieldset, mode, npart=10):
    def MoveRight(particle, fieldset, time, dt):
        fieldset.U[time, particle.lon + 0.1, particle.lat, particle.depth]
        particle.lon += 0.1

    pset = ParticleSet(fieldset, pclass=ptype[mode],
                       lon=np.linspace(0, 1, npart, dtype=np.float32),
                       lat=np.linspace(1, 0, npart, dtype=np.float32))
    error_thrown = False
    try:
        pset.execute(MoveRight, starttime=0., endtime=10., dt=1.)
    except OutOfBoundsError:
        error_thrown = True
    assert error_thrown
    assert len(pset) == npart
    assert (np.array([p.lon - 1. for p in pset]) > -1.e12).all()


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_execution_recover_out_of_bounds(fieldset, mode, npart=2):
    def MoveRight(particle, fieldset, time, dt):
        fieldset.U[time, particle.lon + 0.1, particle.lat, particle.depth]
        particle.lon += 0.1

    def MoveLeft(particle, fieldset, time, dt):
        particle.lon -= 1.

    lon = np.linspace(0.05, 0.95, npart, dtype=np.float32)
    lat = np.linspace(1, 0, npart, dtype=np.float32)
    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=lon, lat=lat)
    pset.execute(MoveRight, starttime=0., endtime=10., dt=1.,
                 recovery={ErrorCode.ErrorOutOfBounds: MoveLeft})
    assert len(pset) == npart
    assert np.allclose([p.lon for p in pset], lon, rtol=1e-5)
    assert np.allclose([p.lat for p in pset], lat, rtol=1e-5)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_execution_delete_out_of_bounds(fieldset, mode, npart=10):
    def MoveRight(particle, fieldset, time, dt):
        fieldset.U[time, particle.lon + 0.1, particle.lat, particle.depth]
        particle.lon += 0.1

    def DeleteMe(particle, fieldset, time, dt):
        particle.delete()

    lon = np.linspace(0.05, 0.95, npart, dtype=np.float32)
    lat = np.linspace(1, 0, npart, dtype=np.float32)
    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=lon, lat=lat)
    pset.execute(MoveRight, starttime=0., endtime=10., dt=1.,
                 recovery={ErrorCode.ErrorOutOfBounds: DeleteMe})
    assert len(pset) == 0


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_kernel_add_no_new_variables(fieldset, mode):
    def MoveEast(particle, fieldset, time, dt):
        particle.lon += 0.1

    def MoveNorth(particle, fieldset, time, dt):
        particle.lat += 0.1

    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=[0.5], lat=[0.5])
    pset.execute(pset.Kernel(MoveEast) + pset.Kernel(MoveNorth),
                 starttime=0., endtime=1., dt=1.)
    assert np.allclose([p.lon for p in pset], 0.6, rtol=1e-5)
    assert np.allclose([p.lat for p in pset], 0.6, rtol=1e-5)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_multi_kernel_duplicate_varnames(fieldset, mode):
    # Testing for merging of two Kernels with the same variable declared
    # Should throw a warning, but go ahead regardless
    def MoveEast(particle, fieldset, time, dt):
        add_lon = 0.1
        particle.lon += add_lon

    def MoveWest(particle, fieldset, time, dt):
        add_lon = -0.3
        particle.lon += add_lon

    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=[0.5], lat=[0.5])
    pset.execute(pset.Kernel(MoveEast) + pset.Kernel(MoveWest),
                 starttime=0., endtime=1., dt=1.)
    assert np.allclose([p.lon for p in pset], 0.3, rtol=1e-5)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_multi_kernel_reuse_varnames(fieldset, mode):
    # Testing for merging of two Kernels with the same variable declared
    # Should throw a warning, but go ahead regardless
    def MoveEast1(particle, fieldset, time, dt):
        add_lon = 0.2
        particle.lon += add_lon

    def MoveEast2(particle, fieldset, time, dt):
        particle.lon += add_lon  # NOQA - no flake8 testing of this line

    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=[0.5], lat=[0.5])
    pset.execute(pset.Kernel(MoveEast1) + pset.Kernel(MoveEast2),
                 starttime=0., endtime=1., dt=1.)
    assert np.allclose([p.lon for p in pset], [0.9], rtol=1e-5)  # should be 0.5 + 0.2 + 0.2 = 0.9


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_update_kernel_in_script(fieldset, mode):
    # Testing what happens when kernels are updated during runtime of a script
    # Should throw a warning, but go ahead regardless
    def MoveEast(particle, fieldset, time, dt):
        add_lon = 0.1
        particle.lon += add_lon

    def MoveWest(particle, fieldset, time, dt):
        add_lon = -0.3
        particle.lon += add_lon

    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=[0.5], lat=[0.5])
    pset.execute(pset.Kernel(MoveEast),
                 starttime=0., endtime=1., dt=1.)
    pset.execute(pset.Kernel(MoveWest),
                 starttime=1., endtime=2., dt=1.)
    assert np.allclose([p.lon for p in pset], 0.3, rtol=1e-5)  # should be 0.5 + 0.1 - 0.3 = 0.3
