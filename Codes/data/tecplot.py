import os
import numpy as np
from typing import Union, List, Tuple
import torch

def read_tec_file(file_name):
    # Read file
    if not os.path.isfile(file_name):
        raise ValueError("File name does not exist.")
        return
    file = open(file_name, 'r')
    file_str = file.read()
    file.close()
    # Prepare zone data and read header
    zones = file_str.split('ZONE T=')
    header = zones.pop(0)
    header = header.replace('TITLE',' ')\
                    .replace('VARIABLES',' ')\
                    .replace('"',' ')\
                    .replace('=',' ')\
                    .replace('\n',' ')\
                    .split(' ')
    variables = list(filter(None, header))
    # Read each zone data
    zone_data = []
    for zone in zones:
        node, element = read_tec_zone('ZONE T='+zone)
        zone_data.append({'node': node, 'element': element})
    return zone_data, variables

def read_tec_zone(s):
    s = s.split(')')
    header = s[0] + ')'
    data = s[1]
    # Process header
    zone_propeties = {}
    for ss in header.replace(',','\n').split('\n'):
        ss = ss.replace(' ','')\
                .split('=')
        zone_propeties[ss[0]]= ss[1]
    n_nodes = int(zone_propeties['Nodes'])
    n_elements = int(zone_propeties['Elements'])
    n_variables = int((len(zone_propeties['DT']) - 2) / len('SINGLE'))
    # Process data
    data = data.replace('\n',' ').split(' ')
    data = list(filter(None, data))
    if zone_propeties['DATAPACKING'] == 'BLOCK':
        node_data = np.array(data[0:n_nodes*n_variables], dtype=np.float64).reshape(n_variables, n_nodes)
        element_data = np.array(data[n_nodes*n_variables:n_nodes*n_variables+n_elements*4], dtype=np.uint32).reshape(n_elements, 4)
    if zone_propeties['DATAPACKING'] == 'POINT':
        node_data = np.array(data[0:n_nodes*n_variables], dtype=np.float64).reshape(n_nodes, n_variables)
        element_data = np.array(data[n_nodes*n_variables:n_nodes*n_variables+n_elements*4], dtype=np.uint32).reshape(n_elements, 4)
    return node_data, element_data

def read_multi_tec_files_concat(file_names, expand_dims=True, axis=-1):
    # Read multiple tecplot file which zones have the same geometry
    # then concatenate all zones into a single array of node and element
    file_name = file_names.pop(0)
    file_data, variables = read_tec_file(file_name)
    for file_name in file_names:
        _file_data, _ = read_tec_file(file_name)
        file_data += _file_data
    data = {}
    for key in file_data[0]:
        data[key] = []
    for i in range(len(file_data)):
        for key in file_data[i]:
            if expand_dims:
                data[key].append(np.expand_dims(file_data[i][key], axis=axis))
            else:
                data[key].append(file_data[i][key])
    for key in data:
        data[key] = np.concatenate(data[key], axis=axis)
    return data, variables

def extract_x_y_from_tec_data(
    data : np.ndarray,
    variable_dim : int,
    variable_index : Union[List, np.ndarray],
    timestep_dim : int,
    timestep_x : int,
    timestep_y : int,
    _combine_dims : bool
):
    """
    Process multi-zones tecplot data.
    Data shape: (n_nodes, n_timesteps, n_variables)
    Extract data from a list of variables, 
    """
    ##
    data = data.take(indices=variable_index, axis=variable_dim)
    ##
    n_timesteps = data.shape[timestep_dim]
    x = []
    y = []
    for i in range(timestep_x, n_timesteps - timestep_y):
        x.append(np.expand_dims(data.take(indices=list(range(i-timestep_x,i)), axis=timestep_dim), axis=-1))
        y.append(np.expand_dims(data.take(indices=list(range(i,i+timestep_y)), axis=timestep_dim), axis=-1))
    x = np.concatenate(x, axis=-1)
    y = np.concatenate(y, axis=-1)
    ##
    if _combine_dims:
        x = combine_dims(x, (variable_dim, timestep_dim))
        y = combine_dims(y, (variable_dim, timestep_dim))
    return x, y

def combine_dims(
    data : np.ndarray,
    dims : Union[Tuple, List, np.ndarray]
):
    # Combine several axes in a numpy array
    shape = list(data.shape)
    dims = list(dims)
    transposed_shape = dims
    combined_shape = [1]
    for i in range(len(dims)):
        if dims[i] >= len(shape):
            raise ValueError("Axes don't match array")
        combined_shape[0] *= shape[dims[i]]
    for i in reversed(range(len(shape))):
        if not (i in dims):
            combined_shape.insert(0, shape[i])
            transposed_shape.insert(0, i)

    transposed_shape = tuple(transposed_shape)
    combined_shape = tuple(combined_shape)
    
    data = data.transpose(transposed_shape)
    data = data.reshape(combined_shape)
    return data

def load_tecplot_to_pt_dataset(
    file_names : List[str],

):
    ##
    data, variables = read_multi_tec_files_concat(
        file_names=file_names,
        expand_dims=True,
        axis=-1
    )
    ##
    x, y = extract_x_y_from_tec_data(
        data=data['node'],
        variable_dim=0,
        variable_index=[3,4,5],
        timestep_dim=2,
        timestep_x=5,
        timestep_y=1,
        _combine_dims=True
    )
    x = np.transpose(x, (1,2,0))
    y = np.transpose(y, (1,2,0))
    ##
    x = torch.tensor()