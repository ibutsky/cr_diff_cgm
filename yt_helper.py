import yt
from yt.frontends.gizmo.api import GizmoDataset
from yt import YTArray
from yt import YTQuantity

def _number_density(field, data):
    # first calculate hydrogen, helium, and metal number densities
    rho = data[('gas', 'density')]

    mh = YTQuantity(1.673557e-24, 'g')
    X = data[('gas', 'H_nuclei_density')] * mh / rho
    Y = data[('gas', 'He_nuclei_mass_density')] / rho
    Z = data[('gas', 'metallicity')]
    mu = 1.0 / (X + 0.25*Y + Z/15.5)

    return rho / (mu * mh)
    
def _cosmic_ray_pressure(field, data):
    cre = data.ds.arr(data[('PartType0', 'CosmicRayEnergy')], 'code_pressure')
#    rho = data[('gas', 'density')]
    return (cre / 3.0).in_units('eV / cm**3')

#def _pressure(field, data):
#    p = data.ds.arr(data[('PartType0', 'Pressure')], 'code_pressure')
#    return p.in_units('eV / cm**3')

def _pressure(field, data):
    te = data[('gas', 'thermal_energy')] * data[('gas', 'density')]
    p = (2./3.) * te
    return p.in_units('eV/cm**3')

def load_ds(model, output = 600, cr = False):
    # current options = 1) m12i_res7100 output 465, 2) cr_700
    # hardcode for now: 
 #   if model == 'm12i':
      #  model = "m12i_res7100"
#        output = 465
      #  model = 'm12i_mass700_MHDCR_tkFIX'
    if model == 'cr':
        model = 'cr_700'
        output = 600

    fn = '/Users/irynabutsky/simulations/FIRE/%s/snapdir_%i/snapshot_%i.0.hdf5'%(model, output, output)
    ds = GizmoDataset(fn)
    ds.add_field(('gas', 'number_density'), function = _number_density, sampling_type = 'particle',
                 display_name = 'Number Density', units = 'cm**-3')
    ds.add_field(('gas', 'pressure'), function=_pressure, sampling_type = 'particle',
                   display_name = 'Gas Pressure', units = 'eV / cm**3')
    if cr or model.__contains__('cr') or model.__contains__('MHDCR'):
        ds.add_field(('gas', 'cosmic_ray_pressure'), function = _cosmic_ray_pressure, sampling_type = 'particle',
                     display_name = 'Cosmic Ray Pressure', units = 'eV / cm**3')

    # find center (need to find better way)
    v, center = ds.find_max(('gas', 'density'))
    return ds, center
