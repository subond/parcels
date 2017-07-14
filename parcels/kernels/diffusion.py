from parcels import rng as random
import math


__all__ = ['BrownianMotion2DFieldKh', 'SpatiallyVaryingDiffusion2D']


def BrownianMotion2DFieldKh(particle, fieldset, time, dt):
    # Kernel for simple Brownian particle diffusion in zonal and meridional direction.
    # Assumes that fieldset has fields Kh_zonal and Kh_meridional

    r = 1/3.
    kh_meridional = fieldset.Kh_meridional[time, particle.lon, particle.lat, particle.depth]
    particle.lat += random.uniform(-1., 1.)*math.sqrt(2*dt*kh_meridional/r)
    kh_zonal = fieldset.Kh_zonal[time, particle.lon, particle.lat, particle.depth]
    particle.lon += random.uniform(-1., 1.)*math.sqrt(2*dt*kh_zonal/r)


def SpatiallyVaryingDiffusion2D(particle, fieldset, time, dt):
    # Diffusion equations for particles in non-uniform diffusivity fields
    # from Ross &  Sharples 2004 and Spagnol et al. 2002
    to_lat = 1 / 1000. / 1.852 / 60.
    to_lon = to_lat / math.cos(particle.lat*math.pi/180)
    r_var = 1/3.
    Rx = random.uniform(-1., 1.)
    Ry = random.uniform(-1., 1.)
    dKdx, dKdy = (fieldset.dK_dx[time, particle.lon, particle.lat], fieldset.dK_dy[time, particle.lon, particle.lat])
    Kfield = fieldset.K[time, particle.lon, particle.lat]
    Rx_component = Rx * math.sqrt(2 * Kfield * dt / r_var) * to_lon
    Ry_component = Ry * math.sqrt(2 * Kfield * dt / r_var) * to_lat
    # Deterministic 'boost' out of areas of low diffusivity
    CorrectionX = dKdx * dt * to_lon
    CorrectionY = dKdy * dt * to_lat
    # diffuse particle
    particle.lon += Rx_component + CorrectionX
    particle.lat += Ry_component + CorrectionY
