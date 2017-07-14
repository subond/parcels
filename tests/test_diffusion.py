from parcels import (FieldSet, ParticleSet, BrownianMotion2DFieldKh, JITParticle, ScipyParticle,
                     GeographicPolarSquare, GeographicSquare)
from parcels import rng as random
from datetime import timedelta as delta
from scipy import stats
import numpy as np
import pytest


ptype = {'scipy': ScipyParticle, 'jit': JITParticle}


def CreateLinearDiffusionField(mesh='spherical', xdim=200, ydim=200):
    """Generates a non-uniform diffusivity field with a linear gradient in one direction"""
    mesh_conversion = 0.01 if mesh is 'spherical' else 1000
    lon = np.linspace(-30*mesh_conversion, 30*mesh_conversion, xdim, dtype=np.float32)
    lat = np.linspace(-60*mesh_conversion, 60*mesh_conversion, ydim, dtype=np.float32)

    Kh = np.zeros((xdim, ydim), dtype=np.float32)
    for x in range(xdim):
        Kh[x, :] = np.tanh(lon[x]/lon[-1]*10.)*xdim/2.+xdim/2.
    dimensions = {'lon': lon, 'lat': lat}
    data = {'U': np.zeros((xdim, ydim), dtype=np.float32),
            'V': np.zeros((xdim, ydim), dtype=np.float32),
            'Kh_zonal': Kh, 'Kh_meridional': Kh}
    fieldset = FieldSet.from_data(data, dimensions, mesh=mesh)

    if mesh is 'spherical':
        fieldset.Kh_zonal.data_converter = GeographicPolarSquare()
        fieldset.Kh_meridional.data_converter = GeographicSquare()

    return fieldset


@pytest.mark.parametrize('mesh', ['spherical', 'flat'])
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_linearKh_Brownian(mesh, mode):
    fieldset = CreateLinearDiffusionField(mesh=mesh)
    npart = 100
    random.seed(1234)
    pset = ParticleSet(fieldset=fieldset, pclass=ptype[mode],
                       lon=np.zeros(npart), lat=np.zeros(npart))
    pset.execute(pset.Kernel(BrownianMotion2DFieldKh), endtime=delta(days=1), dt=delta(minutes=5))
    assert(abs(stats.skew([p.lon for p in pset]) / stats.skew([p.lat for p in pset])) > 10)
