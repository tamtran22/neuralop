import os
import numpy as np
from typing import Union, List, Tuple, Optional
import torch
from neuralop.data.datasets.pt_dataset import PTDataset

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
    root_dir : str,
    dataset_name : str,
    n_train : int, 
    n_test : int, 
    resolution : int, 
    batch_size : int,
    normalize : bool,
    file_names : Optional[List[str]] = None,
):
    # Check if pt file is already exist
    if not os.path.isdir(root_dir):
        raise ValueError('Data directory does not exist.')
    if not os.path.isfile(f'{root_dir}/{dataset_name}_train_{resolution}.pt'):
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
        if normalize:
            _min = min(x.min(), y.min())
            _max = max(x.max(), y.max())
            x = (x - _min) / (_max - _min)
            y = (y - _min) / (_max - _min)
            torch.save({ 'min' : _min, 'max' : _max}, f'{root_dir}/{dataset_name}_train_{resolution}_min_max.pt')
        ##
        x = torch.tensor(x)
        y = torch.tensor(y)
        _data = {'x' : x, 'y' : y}
        torch.save(_data, f'{root_dir}/{dataset_name}_train_{resolution}.pt')
    
    #
    dataset = PTDataset(
        root_dir=root_dir,
        dataset_name = dataset_name,
        n_train = n_train,
        n_tests = [],
        batch_size = batch_size,
        test_batch_sizes = [],
        train_resolution = resolution,
        test_resolutions = [],
        encode_input = False,
        encode_output = False,
        channel_dim=1,
        channels_squeezed = False
    )
    if normalize:
        _min_max = torch.load(f'{root_dir}/{dataset_name}_train_{resolution}_min_max.pt', weights_only=False)
        setattr(dataset, 'min', _min_max['min'])
        setattr(dataset, 'max', _min_max['max'])
    else:
        setattr(dataset, 'min', None)
        setattr(dataset, 'max', None)
    return dataset

def recurrent_formulation(
        
    model,
    initial_input : torch.Tensor,
    n_iteration : int,
    n_timesteps : int,
    n_variables : int,
    device,
):
    """
    model : pytorch neural network model
    initial_input : pytorch Tensor size(1, n_channels, n_nodes)
                    wherea n_channels = n_variables * n_timesteps
    """
    # variable_dim = 1
    timestep_dim = 2
    model.to(device).eval()
    input = initial_input[0].unsqueeze(0).float().to(device)
    recurrent_output = []
    for _ in range(n_iteration):

        output = model(input)

        recurrent_output.append(output.unsqueeze(timestep_dim))

        input = torch.reshape(input, (1, n_variables, n_timesteps, -1))

        _indices = torch.tensor(list(range(1,input.size(timestep_dim)))).to(device)
        input = torch.index_select(input, dim=timestep_dim, index=_indices)

        input = torch.cat([
            input,
            output.detach().unsqueeze(timestep_dim)
        ], dim=timestep_dim)
        input = torch.reshape(input, (1, n_variables * n_timesteps, -1))
    recurrent_output = torch.cat(recurrent_output, dim=timestep_dim)
    return recurrent_output

def write_data_to_tec_file(data, file_name):
    #
    file = open(file_name, 'w+')
    #
    file.write('TITLE     = ""\n')
    file.write('VARIABLES = "x"\n')
    file.write('"y"\n')
    file.write('"z"\n')
    file.write('"u"\n')
    file.write('"v"\n')
    file.write('"w"\n')
    file.write('"p"\n')
    file.write('"f"\n')
    file.write('"vmag"\n')
    file.write('ZONE T="Slice: Arbitrary, Dist=0"\n')
    file.write('STRANDID=2, SOLUTIONTIME=2.4\n')
    file.write('Nodes=78083, Elements=116949, ZONETYPE=FEQuadrilateral\n')
    file.write('DATAPACKING=BLOCK\n')
    file.write('DT=(SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE SINGLE )\n')
    #
    file.close()