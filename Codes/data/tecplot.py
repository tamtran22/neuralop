import os
import numpy as np
from typing import Union, List

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
    node_data = np.array(data[0:n_nodes*n_variables], dtype=np.float64).reshape(n_variables, n_nodes)
    element_data = np.array(data[n_nodes*n_variables:n_nodes*n_variables+n_elements*4], dtype=np.uint32).reshape(n_elements, 4)
    return node_data, element_data

def read_multi_tec_files_concat(file_names, expand_dims=True, axis=-1):
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