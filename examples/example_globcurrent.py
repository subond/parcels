from parcels import FieldSet, ParticleSet, ScipyParticle, JITParticle, AdvectionRK4
from datetime import timedelta as delta
from os import path
from glob import glob
import numpy as np
import pytest


ptype = {'scipy': ScipyParticle, 'jit': JITParticle}


def set_globcurrent_fieldset(filename=None, indices={}):
    if filename is None:
        filename = path.join(path.dirname(__file__), 'GlobCurrent_example_data',
                             '20*-GLOBCURRENT-L4-CUReul_hs-ALT_SUM-v02.0-fv01.0.nc')
    filenames = {'U': filename, 'V': filename}
    variables = {'U': 'eastward_eulerian_current_velocity', 'V': 'northward_eulerian_current_velocity'}
    dimensions = {'lat': 'lat', 'lon': 'lon', 'time': 'time'}
    return FieldSet.from_netcdf(filenames, variables, dimensions, indices)


def test_globcurrent_fieldset():
    fieldset = set_globcurrent_fieldset()
    assert(fieldset.U.lon.size == 81)
    assert(fieldset.U.lat.size == 41)
    assert(fieldset.U.data.shape == (365, 41, 81))
    assert(fieldset.V.lon.size == 81)
    assert(fieldset.V.lat.size == 41)
    assert(fieldset.V.data.shape == (365, 41, 81))

    indices = {'lon': [5], 'lat': range(20, 30)}
    fieldsetsub = set_globcurrent_fieldset(indices=indices)
    assert np.allclose(fieldsetsub.U.lon, fieldset.U.lon[indices['lon']])
    assert np.allclose(fieldsetsub.U.lat, fieldset.U.lat[indices['lat']])
    assert np.allclose(fieldsetsub.V.lon, fieldset.V.lon[indices['lon']])
    assert np.allclose(fieldsetsub.V.lat, fieldset.V.lat[indices['lat']])


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('dt, substart, subend, lonstart, latstart, irange', [
    (3600., 0, 3, 25, -35, range(3, 9, 1)),
    (-3600., 8, 10, 20, -39, range(7, 2, -1))
])
def test_globcurrent_fieldset_advancetime(mode, dt, substart, subend, lonstart, latstart, irange):
    basepath = path.join(path.dirname(__file__), 'GlobCurrent_example_data',
                         '20*-GLOBCURRENT-L4-CUReul_hs-ALT_SUM-v02.0-fv01.0.nc')
    files = glob(str(basepath))

    fieldsetsub = set_globcurrent_fieldset(files[substart:subend])
    psetsub = ParticleSet.from_list(fieldset=fieldsetsub, pclass=ptype[mode], lon=[lonstart], lat=[latstart])

    fieldsetall = set_globcurrent_fieldset(files[0:10])
    psetall = ParticleSet.from_list(fieldset=fieldsetall, pclass=ptype[mode], lon=[lonstart], lat=[latstart])
    if dt < 0:
        psetsub[0].time = fieldsetsub.U.time[-1]
        psetall[0].time = fieldsetall.U.time[-1]

    for i in irange:
        psetsub.execute(AdvectionRK4, starttime=psetsub[0].time, runtime=delta(days=1), dt=dt)
        fieldsetsub.advancetime(set_globcurrent_fieldset(files[i]))

        psetall.execute(AdvectionRK4, starttime=psetall[0].time, runtime=delta(days=1), dt=dt)

    assert abs(psetsub[0].lon - psetall[0].lon) < 1e-4


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_globcurrent_particles(mode):
    fieldset = set_globcurrent_fieldset()

    lonstart = [25]
    latstart = [-35]

    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=lonstart, lat=latstart)

    pset.execute(AdvectionRK4, runtime=delta(days=1), dt=delta(minutes=5),
                 interval=delta(hours=1))

    assert(abs(pset[0].lon - 23.8) < 1)
    assert(abs(pset[0].lat - -35.3) < 1)


@pytest.mark.xfail(reason="Time extrapolation error expected to be thrown")
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_globcurrent_time_extrapolation_error(mode):
    fieldset = set_globcurrent_fieldset()

    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=[25], lat=[-35])
    starttime = fieldset.U.time[0]-delta(days=1).total_seconds()

    pset.execute(AdvectionRK4, starttime=starttime, runtime=delta(days=1),
                 dt=delta(minutes=5), interval=delta(hours=1))
