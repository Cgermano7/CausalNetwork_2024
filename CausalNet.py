import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import os

import tigramite.data_processing as pp
import tigramite.plotting as tp
from tigramite.causal_effects import CausalEffects
from sklearn.linear_model import LinearRegression

# Function to load .nc files
def load_nc_files(file_paths):
    data = []
    for file_path in file_paths:
        with nc.Dataset(file_path, 'r') as ds:
            tp_var = ds.variables['tp'][:]
            data.append(tp_var)
    return np.concatenate(data, axis=0)

# List of .nc file paths
file_paths = ["TP_197901.nc", "TP_197902.nc", "TP_197903.nc", "TP_197904.nc", "TP_197905.nc", "TP_197911.nc", "TP_197912.nc"]

# Load data from .nc files
data = load_nc_files(file_paths)

# Inspect data shape
print("Data shape:", data.shape)

# Flatten the data to 2D (time, lat * lon)
if data.ndim == 3:
    T, lat, lon = data.shape
    data = data.reshape(T, lat * lon)
else:
    T = data.shape[0]
    lat, lon = 1, 1  # Adjust these values based on actual data if needed

# Simple way to define time axis
start_year = 1979
datatime = np.linspace(start_year, start_year + T/12, T)

# Variable names used throughout
var_names = [f'TP_{i}' for i in range(data.shape[1])]

# Time-bin data and datatime
data, _ = pp.time_bin_with_mask(data, time_bin_length=2, mask=None)
datatime, _ = pp.time_bin_with_mask(datatime, time_bin_length=2, mask=None)

# Smooth-width set to 15 years
cycle_length = 6   # a year in bi-monthly time resolution 
smooth_width = 15 * cycle_length

if smooth_width is not None:
    smoothdata_here = pp.smooth(np.copy(data), smooth_width=smooth_width, kernel='gaussian', residuals=False)
    data_here = pp.smooth(np.copy(data), smooth_width=smooth_width, kernel='gaussian', residuals=True)
else:
    print("Not smoothed.")
    data_here = np.copy(data)

# Remove seasonal mean and divide by seasonal standard deviation
def anomalize(dataseries, divide_by_std=True, reference_bounds=None, cycle_length=12, return_cycle=False):
    if reference_bounds is None:
        reference_bounds = (0, len(dataseries))

    anomaly = np.copy(dataseries)
    for t in range(cycle_length):
        if return_cycle:
            anomaly[t::cycle_length] = dataseries[t + reference_bounds[0]:reference_bounds[1]:cycle_length].mean(axis=0)
        else:
            anomaly[t::cycle_length] -= dataseries[t + reference_bounds[0]:reference_bounds[1]:cycle_length].mean(axis=0)
            if divide_by_std:
                anomaly[t::cycle_length] /= dataseries[t + reference_bounds[0]:reference_bounds[1]:cycle_length].std(axis=0)
    return anomaly

seasonal_cycle = anomalize(np.copy(data_here), cycle_length=cycle_length, return_cycle=True)
smoothdata_here += seasonal_cycle

# Create mask for neutral phases and certain months
mask = np.ones(data.shape, dtype='bool')
for i in [0, 5]:
    mask[i::cycle_length, :] = False

# Dataframe for raw data
raw_dataframe = pp.DataFrame(np.copy(data), mask=mask, var_names=var_names, datatime=datatime)

# Dataframe for smoothed data
smoothdataframe_here = pp.DataFrame(smoothdata_here, var_names=var_names,  datatime=datatime)

fig, axes = tp.plot_timeseries(
        raw_dataframe,
        figsize=(8, 5),
        grey_masked_samples='data',
        color='black',
        show_meanline=True,
        adjust_plot=False,
        )  

tp.plot_timeseries(
        smoothdataframe_here,
        fig_axes = (fig, axes),
        grey_masked_samples='data',
        show_meanline=False,
        color='red',
        alpha=0.4,
        adjust_plot=True,
        tick_label_size=7,
        label_fontsize=8,
        time_label='year',
        var_units=['m'],
#         save_name="timeseries.pdf"
        ); plt.show()


if smooth_width is not None:
    data_here = pp.smooth(data=np.copy(data), smooth_width=smooth_width, kernel='gaussian', residuals=True)
else:
    data_here = np.copy(data)

data_here = anomalize(data_here, cycle_length=cycle_length)

# Initialize Tigramite dataframe with mask, missing_flag is not needed here
dataframe = pp.DataFrame(data_here, mask=mask, var_names=var_names, missing_flag=999.)

# Causal effect estimation
graph = np.array([
        [['', '', '']],
        [['-->', '']],
        [['<--', '-->']],
        [['', '-->']]
], dtype='<U3')

# Positions of nodes for process graph
node_pos =  {
            'y': np.array([0.5, 0., 0.5, 1.]),
            'x': np.array([0., 0.5, 1., .5])
            }

# Show both graphs next to each other
fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(6, 2.5))

tp.plot_graph(
    fig_ax=(fig, axes[0]),
    graph=graph,
    node_pos=node_pos,
    arrow_linewidth=5,
    node_size=0.2,
    node_aspect=1.5,
    var_names=var_names,
    tick_label_size=6,
    )
axes[0].set_title('Process graph', pad=20)

tp.plot_time_series_graph(
    fig_ax=(fig, axes[1]),
    graph=graph,
    var_names=var_names,
    )
axes[1].set_title('Time series graph', pad=20)

fig.tight_layout()

# Define cause X and effect Y
X = [(0, 0)]
Y = [(1, 0)]

# Initialize class with stationary directed acyclic graph (DAG) defined above
causal_effects = CausalEffects(graph, graph_type='stationary_dag', X=X, Y=Y, S=None, verbosity=0)

print("X = %s -----> Y = %s" % (str([(var_names[var[0]], var[1]) for var in X]), str([(var_names[var[0]], var[1]) for var in Y])))

# Get optimal adjustment set
opt = causal_effects.get_optimal_set()

if not opt:
    print("NOT IDENTIFIABLE!")
print("Oset = ", [(var_names[v[0]], v[1]) for v in opt])

# Color nodes
special_nodes = {}
for node in opt:
    special_nodes[node] = 'orange'
for node in causal_effects.M:
    special_nodes[node] = 'lightblue'
for node in causal_effects.X:
    special_nodes[node] = 'red'
for node in causal_effects.Y:
    special_nodes[node] = 'blue'
    
fig, ax = tp.plot_time_series_graph(
        graph=graph,
        special_nodes=special_nodes,
        var_names=var_names,
        figsize=(3, 2.5),
        )
plt.tight_layout()

# Optional data transform
data_transform = None  # sklearn.preprocessing.StandardScaler()

# Confidence interval range
conf_lev = 0.9

# Fit causal effect model from observational data
causal_effects.fit_total_effect(
    dataframe=dataframe, 
    mask_type='y',
    estimator=LinearRegression(),
    data_transform=data_transform,
    )

# Fit bootstrap causal effect model
causal_effects.fit_bootstrap_of(
    method='fit_total_effect',
    method_args={'dataframe':dataframe,  
    'mask_type':'y',
    'estimator':LinearRegression(),
    'data_transform':data_transform,
    },
    seed=4
    )

# Define interventions
dox_vals = np.linspace(0., 1., 2)

# Predict effect of interventions do(X=0.), ..., do(X=1.) in one go
intervention_data = np.repeat(dox_vals.reshape(len(dox_vals), 1), axis=1, repeats=len(X))
pred_Y = causal_effects.predict_total_effect(intervention_data=intervention_data)

# Bootstrap: Predict effect of interventions do(X=0.), ..., do(X=1.) in one go
intervention_data = np.repeat(dox_vals.reshape(len(dox_vals), 1), axis=1, repeats=len(X))
conf = causal_effects.predict_bootstrap_of(
    method='predict_total_effect',
    method_args={'intervention_data':intervention_data},
    conf_lev=conf_lev)

print("Total effect via adjustment = %.2f [%.2f, %.2f]"
              %(pred_Y[1]-pred_Y[0], conf[0,1]-conf[0,0], conf[1,1]-conf[1,0])) 
