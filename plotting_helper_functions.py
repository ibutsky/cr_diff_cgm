import yt
from yt.frontends.gizmo.api import GizmoDataset
from yt import YTArray, YTQuantity

import numpy as np
import h5py as h5
import os
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm, ListedColormap
import seaborn as sns
import palettable 
sns.set_style("ticks",{'axes.grid': True, 'grid.linestyle': '--'})
import yt_helper as yth

def generate_projection_data(model, output = 600, field_list = [],
                             weight_list = [], radius = 1000, resolution = 800):

    plot_data = h5.File('data/projection_data_%s_%ikpc.h5'%(model, radius), 'a')

    existing_keys = list(plot_data.keys())

    all_fields_already_generated = True
    for field in field_list:
        for orientation in ['x', 'y', 'z']:
            if '%s_%s'%(field[1], orientation) not in existing_keys:
                all_fields_already_generated = False
    if all_fields_already_generated:
        return
    
    ds, cen = yth.load_ds(model)
#    sp = ds.sphere(cen, (radius, 'kpc'))
    half_length = YTArray(np.array([radius, radius, radius]), 'kpc')
    
    left_edge = cen - half_length#ds.arr(cen.in_units('kpc') - ds.arr(half_length, 'kpc') YTArray(0.5*[width, width, width], 'kpc')
    right_edge = cen + half_length#YTArray(0.5*[width, width, width], 'kpc')
    print(left_edge, right_edge)
    box = ds.region(cen, left_edge, right_edge)

    # set up projection plots for fields that are weighted and unweighted                                                         
    #del plot_data['radial_velocity']   
    if 'radius' not in plot_data.keys():
        width = yt.YTQuantity(radius*2, 'kpc')
        px, py = np.mgrid[-width/2:width/2:resolution*1j, -width/2:width/2:resolution*1j]
        radius = (px**2.0 + py**2.0)**0.5
        plot_data.create_dataset("radius", data = radius.ravel())
    
    for view in ['x', 'y', 'z']:
        for i in range(len(field_list)):
            print(view, field_list[i])
            field = field_list[i][1]
            dset = '%s_%s'%(field,view)
            if dset not in plot_data.keys():
                proj = yt.ProjectionPlot(ds, view, field, weight_field = weight_list[i], 
                                         width=width, center = cen, data_source = box)
                proj_frb =  proj.data_source.to_frb(width, resolution)

                plot_data.create_dataset(dset, data = np.array(proj_frb[field_list[i]]))
                plot_data.flush()
                

def find_plot_axis_range(data, log = False, gap = 0.1):
    data_max = np.max(data)
    data_min = np.min(data)
    if log:
        data_range = np.log10(data_max/data_min)
        data_lims = (np.log10(data_min) - gap*data_range,
                     np.log10(data_max) + gap*data_range)
    else:
        data_range = data_max - data_min
        data_lims = (data_min - gap*data_range, data_max + gap*data_range)
    return data_lims
    

def get_default_units(field):
    if np.size(field) > 1:
        field = field[1]
        
    if field.__contains__('radius'):
        unit = 'kpc'
    elif field.__contains__('mass'):
        unit = 'Msun'
    elif field == 'ones':
        unit = ''
    elif field.__contains__('pressure'):
        unit = 'eV / cm**3'
    else:
        unit == ''
    return unit

def get_default_limits(field):
    if np.size(field) > 1:
        field = field[1]

    elif field == 'ones':
        unit = ''
    elif field.__contains__('pressure'):
        unit = 'eV / cm**3'
    else:
        unit == ''
    return unit


def generate_profile_from_projection(field, model, radius = 1000, resolution = 800, 
                                     xlog = False, ylog = True, ylims = None, nbins = None, pressure_lims = (1e-6, 1e2)):

    frb = h5.File('data/projection_data_%s_%ikpc.h5'%(model, radius), 'r')
    r_arr = np.array([]) # spatial information measuring distance from center of plot
    img_arr = np.array([]) # the plot data from the projection plot
    for axis in ['x', 'y', 'z']:
        cname = "%s_%s"%(field, axis)
        if cname in frb.keys():
            r_arr = np.concatenate((r_arr, frb['radius'][:]))
            img_arr = np.concatenate((img_arr, frb[cname][:].ravel()))
            
    if nbins is None:
        nbins = int(resolution / 2)

    xbins = np.linspace(0, radius, nbins)
    if xlog:
        xbins = np.logspace(1, np.log10(radius),nbins)
#    if ylims is None:
#        ylims = find_plot_axis_range(img_arr, log = ylog)
#    ylims = (-6, 2)
    ybins = np.linspace(ylims[0], ylims[1], nbins)
    if ylog:
        ybins = np.power(10, ybins)
        
        
    counts, x_edge, y_edge = np.histogram2d(r_arr, img_arr, bins=(xbins, ybins))
    x_bin_center = ((x_edge[1:] + x_edge[:-1]) / 2).reshape(nbins-1,1)
    # normalize counts in x-space to remove out linear increase in counts with                                                
    # radius due to circles of constant impact parameter                                                                      
    counts /= x_bin_center
    return xbins, ybins, counts



def calculate_median_profile_from_meshgrid(x, y, z, confidence = 0.95, nbins = 100, centered = True, 
                                           convert_to_linear = True):
    # assuming z, x, y are the outputs of np.histogram2d
    if len(x) > len(z):
        x = x[:-1]
    if len(y) > len(z):
        y = y[:-1]
    
    xbins           = np.linspace(np.min(x), np.max(x), nbins+1)
    centered_x_bins = (xbins + (np.max(x)/nbins/2.0))
    
    median          = np.zeros(nbins)
    mean            = np.zeros(nbins)
    lowlim          = np.zeros(nbins)
    uplim           = np.zeros(nbins)
    std             = np.zeros(nbins)
    for i in range(nbins):
        mask = (x >= xbins[i]) & (x < xbins[i+1])
        sample        = np.ndarray.sum(z[mask], axis = 0)
        sum_sample    = np.sum(sample)
        cumsum_sample = np.cumsum(sample)
        median_index = np.where(cumsum_sample > 0.5 * sum_sample)[0]
        lowlim_index = np.where(cumsum_sample > (1.0 - confidence)*sum_sample)[0]
        uplim_index  = np.where(cumsum_sample > (confidence)*sum_sample)[0]
        if median_index.size != 0:
            median[i] = y[median_index[0]]
            mean[i]   = np.average(y, weights = sample)#np.mean(y*sample) / sum_sample
            
            lowlim[i] = y[lowlim_index[0]]#low_array[0]]
            uplim[i]  = y[uplim_index[0]]# up_array[0]]
            std[i]    = np.std(y*sample) / sum_sample#, weights = sample)
    if centered:
        xbins = centered_x_bins
        
    if convert_to_linear:
        xbins = np.power(10, xbins)
        median = np.power(10, median)
        std_lower = np.power(10, mean - 2*std)
        std_upper = np.power(10, mean + 2*std)
        mean = np.power(10, mean)
        lowlim = np.power(10, lowlim)
        uplim = np.power(10, uplim)
    else:
        std_lower = mean - std
        std_upper = mean + std
    return xbins[:-1], median, mean, lowlim, uplim #, std_lower, std_upper


def generate_radial_pressure_profile_data(h5file, field_list, model, xfield = 'spherical_position_radius', 
                                          weight_field = 'mass', nbins = 500, pressure_units = 'eV/cm**3', extent = 1000):
    
    # assumption is: if this routine is called, it means the data for this field hasn't been generated yet
    
    ds, center = yth.load_ds(model)
    sp = ds.sphere(center, (extent, 'kpc'))
    
    xdata = np.log10(sp[('gas', xfield)].in_units('kpc'))  # assuming xfield is some sort of radius field
    xdata[xdata==0] = -10 # in theory this should never be zero. in practice, it happens and breaks everything
    xbins = np.linspace(1, np.log10(extent), nbins)
    ybins = np.linspace(-6, 3, nbins)
    
    zdata = sp[('gas', weight_field)].in_units(get_default_units(weight_field))
    
    #empty dictionary
   # profile_data = {}#dict.fromkeys(field_list, np.array([]))
    
    for i, field in enumerate(field_list):
        ydata = np.log10(sp[('gas', field)].in_units(pressure_units))

        H, xedges, yedges = np.histogram2d(xdata, ydata, bins = (xbins, ybins), weights = zdata)
#        plt.pcolormesh(H, xedges, yedges)
#        plt.savefig('test.png')
        xbins, median, mean, lowlim, uplim = calculate_median_profile_from_meshgrid(xedges[:-1], yedges[:-1], H)
       
        h5file.create_dataset('%s_median'%field, data = median)
        h5file.create_dataset('%s_mean'%field,   data = mean)
        h5file.create_dataset('%s_lowlim'%field, data = lowlim)
        h5file.create_dataset('%s_uplim'%field,  data = uplim)
        

    # just in case this wasn't already loaded:
    if xfield not in list(h5file.keys()):
        h5file.create_dataset(xfield, data = xbins)
        

        
                                      
                                      

def get_radial_pressure_profile_data(model, field_list = [], xfield = 'spherical_position_radius', 
                                        weight_field = 'mass', data_dir = 'data'):
    
    #sim_location = get_sim_location(model, resolution, sim_dir = sim_dir)
    if weight_field == 'ones':
        weight_type = 'volume'
    else:
        weight_type = 'mass'
    stored_data_file = h5.File('%s/radial_pressure_profile_data_%s_%s.h5'%(data_dir, weight_type, model), 'a')
    existing_keys = list(stored_data_file.keys())
    
    
    data_to_output = {}#dict.fromkeys(field_list, np.array([]))
    field_list = np.array(field_list) # needs to be numpy array
    
    
    # first, find the fields that aren't already saved
    fields_to_generate = field_list[['%s_median'%field not in existing_keys for field in field_list]]
    
    # generate and save the missing data
    if np.size(fields_to_generate) > 0:
        generate_radial_pressure_profile_data(stored_data_file, fields_to_generate, model,
                                              xfield = xfield, weight_field = weight_field)
        
    # now we should have all the data and can just load it in
    data_to_output[xfield] = np.array(stored_data_file.get(xfield))
    for field in field_list:
        for data_type in ['mean', 'median', 'lowlim', 'uplim']:
            field_entry = '%s_%s'%(field, data_type)
            # copying into dictionary so that we can close h5 file... idk if that's realy necessary
            data_to_output[field_entry] = np.array(stored_data_file.get(field_entry))
        
    stored_data_file.close()
    return data_to_output
   
                                      
def estimate_sfr(model, time_interval = 1e9):
    time, sfr = get_sfh_data(model)

    time *= 1e9 # converting from Gyr to yr
    dt = (time[1::2] - time[::2])[0] # dt defined to be constant throughout
    # destacking, since get_sfh_data originally made to plot sfr
    time = time[1::2] # in units of yr
    sfr = sfr[::2] # in units of Msun/yr

    current_time = time[-1]
    mask = time >= current_time - time_interval

    stellar_mass = np.sum(sfr[mask] * dt) 
    
    return stellar_mass / time_interval


def get_vcirc_profile(model, impact_list = None, radius = 1000, mass_unit = 'g', save = True):
    # V_c = sqrt(GM(<r) /r)
    fname = 'data/vcirc_profile_%s.h5'%model
    if os.path.isfile(fname):
        h5file = h5.File(fname, 'r')
        impact = h5file['impact'][:]
        vc_list = h5file['vcirc'][:]
    else:
        if impact_list is None:
            impact_list = np.logspace(1,np.log10(radius), 100)
            save = False

        ds, center = yth.load_ds(model)
        max_r = 1.2 * np.max(impact_list)
        ad = ds.sphere(center, (max_r, 'kpc'))
    
        G = YTQuantity(6.6743e-11, 'm**3 / kg / s**2')
        r_gas = ad[('gas', 'spherical_position_radius')].in_units('kpc')
        r_star = ad[('PartType4', 'particle_position_spherical_radius')].in_units('kpc')
        r_dark = ad[('PartType1', 'particle_position_spherical_radius')].in_units('kpc')
        m_gas = ad[('gas', 'mass')]#.in_units(mass_unit)
        m_star = ad[('PartType4', 'particle_mass')]#.in_units(mass_unit)
        m_dark = ad[('PartType1', 'particle_mass')]#.in_units(mass_unit)
        # mtot = m_gas + m_star + m_dark
    
        vc_list = np.array([])
        for impact in impact_list:
            impact = YTQuantity(impact, 'kpc')
            # mask = r < impact
            #m_enc = YTQuantity(np.sum(mtot[mask]), mass_unit)
            m_enc = np.sum(m_gas[r_gas <= impact]) + np.sum(m_star[r_star <= impact]) + np.sum(m_dark[r_dark <= impact])
            vc = np.sqrt(G * m_enc / impact).in_units('km/s')
            vc_list = np.append(vc_list, vc)

        if save:
            h5file = h5.File(fname, 'w')
            h5file.create_dataset('impact', data = impact_list)
            h5file.create_dataset('vcirc', data = vc_list)
        
    return impact, vc_list


def get_min_diffusivity_estimate(model, extent = 1000, sfr_time_interval = 1e9):
    fname = 'data/min_diffusivity_%s.h5'%model
    if os.path.isfile(fname):
        h5file = h5.File(fname, 'r')
        impact = h5file['impact'][:]
        min_kappa = h5file['min_kappa'][:]
    else:
        impact, Hcol = get_radial_H_column(model, extent = extent)
        sfr = estimate_sfr(model, time_interval = sfr_time_interval)
        temp, vc = get_vcirc_profile(model, impact)
        min_kappa = 5 * (1e19 / Hcol) * sfr * np.power((200 / vc), 2) * 1e30
        
        h5file = h5.File(fname, 'w')
        h5file.create_dataset('impact', data = impact)
        h5file.create_dataset('min_kappa', data = min_kappa)
        h5file.create_dataset('vc', data = vc)
        
    return np.array(impact), np.array(min_kappa)


def get_radial_H_column(model, extent = 1000):

    field = 'H_nuclei_density'
    fname = 'data/H_col_profile_%s.h5'%model
    if os.path.isfile(fname):
        h5file = h5.File(fname, 'r')
        impact = h5file['impact'][:]
        median = h5file['Hcol_med'][:]
    else:      
        generate_projection_data(model, field_list = [('gas', field)], weight_list = [None], radius = extent)
        xbins, ybins, counts = generate_profile_from_projection('H_nuclei_density', model, radius = extent, ylims = (18, 22))
        impact, median, mean, lowlim, uplim = calculate_median_profile_from_meshgrid(xbins[1:], ybins[1:], counts,
                                                                  nbins = 100, convert_to_linear = False)
        h5file = h5.File(fname, 'w')
        h5file.create_dataset('impact', data = impact)
        h5file.create_dataset('Hcol_med', data = median)
        h5file.create_dataset('Hcol_mean', data = mean)
        h5file.create_dataset('Hcol_lowlim', data = lowlim)
        h5file.create_dataset('Hcol_uplim', data = uplim)
        
    return np.array(impact), np.array(median)


def get_sfh_data(model, nbins = 100):

    fname = 'data/sfh_%s.h5'%model
    if os.path.isfile(fname):
        h5file = h5.File(fname, 'r')
        time = h5file['time'][:]
        sfr = h5file['sfr'][:]
    else:
        ds, center = yth.load_ds(model)
        ad = ds.all_data()
        creation_time = ad[('PartType4', 'creation_time')].in_units('yr')
        stellar_mass = ad[('PartType4', 'particle_mass')].in_units('Msun')
        current_time = ds.current_time.in_units('yr')
    
        time_list = np.linspace(0, current_time, nbins)
        mass_list = np.array([])
        for i in range(1, len(time_list)):
            dt = time_list[i] - time_list[i-1]
            mask = (creation_time >= time_list[i-1]) & (creation_time < time_list[i])
            mass_list = np.append(mass_list, np.sum(stellar_mass[mask]) / dt)
        x = np.vstack((time_list[:-1], time_list[1:])).reshape((-1), order = 'F') / 1e9 # in Gyr
        y = np.vstack((mass_list, mass_list)).reshape((-1), order = 'F')
    
        h5file = h5.File(fname, 'w')
        h5file.create_dataset('time', data = x)
        h5file.create_dataset('sfr', data = y)
    
    return time, sfr
