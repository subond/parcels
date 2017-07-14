from parcels import Field, FieldSet, ParticleSet, BrownianMotion2DFieldKh, JITParticle, ScipyParticle, Variable, SpatiallyVaryingDiffusion2D, random, ErrorCode
from operator import attrgetter
from datetime import timedelta as delta
import matplotlib.pyplot as plt
import numpy as np
import math
from argparse import ArgumentParser
import pytest


def LinearDiffusionField(lon, lat):
    """Generates a non-uniform diffusivity field with a linear gradient in one direction
    """
    Kh = np.zeros((lon.size, lat.size), dtype=np.float32)
    for x in range(lon.size):
        Kh[x, :] = x
    return Field('Kh', Kh, lon, lat, transpose=True)


def CreateZeroAdvectionField(xdim=200, ydim=200):
    dimensions = {'lon': np.linspace(-10000, 10000, xdim, dtype=np.float32),
                  'lat': np.linspace(-10000, 10000, ydim, dtype=np.float32)}

    data = {'U': np.zeros((xdim, ydim), dtype=np.float32),
            'V': np.zeros((xdim, ydim), dtype=np.float32)}

    return FieldSet.from_data(data, dimensions, mesh='flat')


fieldset = CreateZeroAdvectionField()
fieldset.add_field(LinearDiffusionField(fieldset.U.lon, fieldset.U.lat))
npart = 1000
pset = ParticleSet(fieldset=fieldset, pclass=JITParticle,
                   lon=np.zeros(npart), lat=np.zeros(npart))
pset.execute(pset.Kernel(BrownianMotion2DFieldKh), endtime=delta(days=1), dt=delta(hours=1))
print np.mean([p.lon for p in pset])

def CreateStartField(lon, lat):
    time = np.arange(0., 100000, 100000/2., dtype=np.float64)
    # An evenly distributed starting density
    data = np.ones((lon.size, lat.size, time.size), dtype=np.float32)
    return data


def LagrangianDiffusionNoCorrection(particle, fieldset, time, dt):
    # Version of diffusion equation with no determenistic term i.e. brownian motion
    to_lat = 1 / 1000. / 1.852 / 60.
    to_lon = to_lat / math.cos(particle.lat*math.pi/180)
    r_var = 1/3.
    Rx = random.uniform(-1., 1.)
    Ry = random.uniform(-1., 1.)
    Kfield = fieldset.K[time, particle.lon, particle.lat]
    Rx_component = Rx * math.sqrt(2 * Kfield * dt / r_var) * to_lon
    Ry_component = Ry * math.sqrt(2 * Kfield * dt / r_var) * to_lat
    particle.lon += Rx_component
    particle.lat += Ry_component


def UpdatePosition(particle, fieldset, time, dt):
    particle.prev_lon = particle.new_lon
    particle.new_lon = particle.lon
    particle.prev_lat = particle.new_lat
    particle.new_lat = particle.lat


# Recovery Kernal for particles that diffuse outside boundary
def Send2PreviousPoint(particle):
    # print("Recovery triggered at %s | %s!" % (particle.lon, particle.lat))
    # print("Moving particle back to %s | %s" % (particle.prev_lon, particle.prev_lat))
    particle.lon = particle.prev_lon
    particle.lat = particle.prev_lat


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def diffusion_test(mode, type='true_diffusion', particles=1000, timesteps=1000, output_file='diffusion_test'):
    # Generating fieldset with zero horizontal velocities
    forcing_fields = CreateDummyUV(100, 100)
    fieldset = FieldSet(forcing_fields['U'], forcing_fields['V'], forcing_fields['U'].depth,
                        forcing_fields['U'].time, fields=forcing_fields)
    # Create a non-uniform field of diffusivity
    fieldset.add_field(CreateDiffusionField(fieldset))
    # Calculate first differential of diffusivity field (needed for diffusion)
    divK = fieldset.K.gradient()
    fieldset.add_field(divK[0])
    fieldset.add_field(divK[1])
    # Calculate second differential (needed to estimate the appropriate minimum timestep to approximate Eulerian diffusion)
    fieldset.add_field(fieldset.dK_dx.gradient(name="d2K")[0])
    fieldset.add_field(fieldset.dK_dy.gradient(name="d2K")[1])

    # Evenly spread starting distribution
    Start_Field = Field('Start', CreateStartField(fieldset.U.lon, fieldset.U.lat),
                        fieldset.U.lon, fieldset.U.lat, depth=fieldset.U.depth, time=fieldset.U.time, transpose=True)

    timestep = 500

    steps = timesteps

    ParticleClass = JITParticle if mode == 'jit' else ScipyParticle

    # Simply particle superclass that remembers previous positions for kernel error recovery
    class Diffuser(ParticleClass):
        prev_lon = Variable("prev_lon", to_write=False, initial=attrgetter('lon'))
        prev_lat = Variable("prev_lat", to_write=False, initial=attrgetter('lat'))
        new_lon = Variable("new_lon", to_write=False, initial=attrgetter('lon'))
        new_lat = Variable("new_lat", to_write=False, initial=attrgetter('lat'))

    diffusers = fieldset.ParticleSet(size=particles, pclass=Diffuser, start_field=Start_Field)

    # Particle density at simulation start should be more or less uniform
    DensityField = Field('temp', np.zeros((5, 5), dtype=np.float32),
                         np.linspace(np.min(fieldset.U.lon), np.max(fieldset.U.lon), 5, dtype=np.float32),
                         np.linspace(np.min(fieldset.U.lat), np.max(fieldset.U.lat), 5, dtype=np.float32))
    StartDensity = diffusers.density(DensityField, area_scale=False, relative=True)
    fieldset.add_field(Field(type + 'StartDensity', StartDensity,
                             DensityField.lon,
                             DensityField.lat))

    diffuse = diffusers.Kernel(SpatiallyVaryingDiffusion2D) if type == 'true_diffusion' else diffusers.Kernel(LagrangianDiffusionNoCorrection)

    diffusers.execute(diffusers.Kernel(UpdatePosition) + diffuse, endtime=fieldset.U.time[0]+timestep*steps, dt=timestep,
                      output_file=diffusers.ParticleFile(name=args.output+type),
                      interval=timestep, recovery={ErrorCode.ErrorOutOfBounds: Send2PreviousPoint})

    EndDensity = diffusers.density(DensityField, area_scale=False, relative=True)
    fieldset.add_field(Field(type+'FinalDensity', EndDensity,
                             DensityField.lon,
                             DensityField.lat))
    fieldset.write(output_file)

    print(type + ' start variation = %s' % np.var(StartDensity))
    print(type + ' end variation = %s' % np.var(EndDensity))

    return [StartDensity, EndDensity]
