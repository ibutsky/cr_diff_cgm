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
    
    

#def _pressure(field, data):
#    p = data.ds.arr(data[('PartType0', 'Pressure')], 'code_pressure')
#    return p.in_units('eV / cm**3')

def _pressure(field, data):
    kT = data[('gas', 'kT')].in_units('eV')
    n =  data[('gas', 'number_density')].in_units('cm**-3')
    return n*kT


def load_ds(model, output = 600):
    # current options = 1) m12i_res7100 output 465, 2) cr_700
    # hardcode for now: 
    if model == 'm12i':
        model = "m12i_res7100"
        output = 465
    elif model == 'cr':
        model = 'cr_700'
        output = 600
    fn = '/Users/irynabutsky/simulations/FIRE/%s/snapdir_%i/snapshot_%i.0.hdf5'%(model, output, output)
    ds = GizmoDataset(fn)
    ds.add_field(('gas', 'number_density'), function = _number_density, sampling_type = 'particle',
                 display_name = 'Number Density', units = 'cm**-3')
    ds.add_field(('gas', 'pressure'), function=_pressure, sampling_type = 'particle',
                   display_name = 'Gas Pressure', units = 'eV / cm**3')

    # find center (need to find better way)
    v, center = ds.find_max(('gas', 'density'))
    return ds, center
