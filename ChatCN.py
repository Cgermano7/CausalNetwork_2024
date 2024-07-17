import netCDF4 as nc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tigramite.data_processing import DataFrame
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
import networkx as nx
import tigramite.plotting as tp
import os

# Define a function to read NetCDF data
def read_nc_file(file_path, variable_name):
    dataset = nc.Dataset(file_path)
    variable_data = dataset.variables[variable_name][:]
    return variable_data

# List of NetCDF files
nc_files = ["TP_197901.nc", "TP_197902.nc", "TP_197903.nc", "TP_197904.nc", "TP_197905.nc", "TP_197911.nc", "TP_197912.nc"]

# Extract total precipitation data from each file
tp_data = []
for file in nc_files:
    tp_data.append(read_nc_file(file, 'tp'))

# Convert the list to a numpy array and reshape
tp_data = np.concatenate(tp_data, axis=0)

# Optionally, average over latitude and longitude if necessary
tp_data_avg = np.mean(tp_data, axis=(1, 2))

# Create a pandas DataFrame
df = pd.DataFrame(tp_data_avg, columns=['Total Precipitation'])

# Create a Tigramite DataFrame
dataframe = DataFrame(df.values, datatime={0: np.arange(len(df))}, var_names=['Total Precipitation'])

# Initialize ParCorr and PCMCI
parcorr = ParCorr(significance='analytic')
pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=1)

# Run PCMCI to get causal links
tau_max = 10  # Maximum lag
results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=None)

# Print significant links
q_matrix = pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'], fdr_method='fdr_bh')
significant_links = (results['val_matrix'] * (q_matrix <= 0.05).astype(int))

print("Significant Links: \n", significant_links)

# Extract significant links
significant_links = (q_matrix <= 0.05) & (results['val_matrix'] != 0)
link_matrix = np.zeros_like(results['val_matrix'])
link_matrix[significant_links] = results['val_matrix'][significant_links]

# Create a directed graph
G = nx.DiGraph()

# Add nodes for each time lag
for t in range(-tau_max, tau_max + 1):
    G.add_node(f'Total Precipitation (t-{t})')

# Add edges
num_lags = link_matrix.shape[1]  # Number of time lags
for i in range(link_matrix.shape[0]):  # Iterate over variables (only one variable here)
    for j in range(num_lags):
        for k in range(num_lags):
            if significant_links[i, j, k]:
                G.add_edge(f'Total Precipitation (t-{j})', f'Total Precipitation (t-{k})', weight=link_matrix[i, j, k])

# Plot the graph
pos = nx.spring_layout(G)
weights = nx.get_edge_attributes(G, 'weight').values()
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=10, font_weight='bold', arrows=True)
nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f'{d["weight"]:.2f}' for u, v, d in G.edges(data=True)}, font_color='red')
plt.title('Causal Network')
plt.show()

# Plotting the time series data
tp.plot_timeseries(dataframe=dataframe)
plt.show()
