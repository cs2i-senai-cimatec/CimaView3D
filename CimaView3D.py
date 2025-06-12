# CIGVis https://github.com/JintaoLee-Roger/cigvis
# Copyright (c) 2023 Jintao Li
# Computational and Interpretation Group (CIG),
# University of Science and Technology of China (USTC).

# CimaView3D https://github.com/cs2i-senai-cimatec/CimaView3D
# Copyright (c) 2025 Darlan Anschau
# CIMATEC - Salvador - BA - Brazil 

# All rights reserved

# This code is licensed under MIT license (see LICENSE for details)

"""
In plotly, for a seismic volume,
- x means inline order
- y means crossline order
- z means time order

- ni means the dimension size of inline / x
- nx means the dimension size of crossline / y
- nt means the dimension size of time / depth / z

Examples
---------
>>> volume.shape = (192, 200, 240) # (ni, nx, nt)

\# only slices
>>> slice_traces = create_slices(volume, pos=[0, 0, 239], cmap='Petrel', show_cbar=True)
>>> plot3D(slice_traces)

\# add surface

\# surfs = [surf1, surf2, ...], each shape is (ni, nx)
>>> sf_traces = create_surfaces(surfs, surf_color='depth')

\# or use amplitude as color
>>> sf_traces = create_surfaces(surfs, volume, surf_color='amp')
>>> plot3D(slice_traces+sf_traces)

For more and detail examples, please refer our documents
"""

import numpy as np
import cigvis
from cigvis import colormap

from IPython import display

import warnings
from typing import List, Tuple, Union, Dict
import numpy as np
from cigvis import ExceptionWrapper
try:
    import plotly.graph_objects as go
except BaseException as e:
    go = ExceptionWrapper(
        e,
        "run `pip install \"cigvis[plotly]\"` or run `pip install \"cigvis[all]\"` to enable jupyter support"
    )

from skimage.measure import marching_cubes
from skimage import transform

import cigvis
from cigvis import colormap
from cigvis.utils import plotlyutils
import cigvis.utils as utils
from plotly.subplots import make_subplots

def create_slices(volume: np.ndarray,
                  pos: Union[List, Dict] = None,
                  clim: List = None,
                  cmap: str = 'Petrel',
                  scale: float = 1,
                  show_cbar: bool = True,
                  cbar_params: Dict = None,
                  type: str = 'faces',
                  **kwargs):
    """
    Parameters
    ----------
    volume : array-like
        3D array
    pos : List or Dict
        init position of the slices, can be a List or Dict, such as:
        ```
        pos = [0, 0, 200] # x: 0, y: 0, z: 200
        pos = [[0, 200], [9], []] # x: 0 and 200, y: 9, z: None
        pos = {'x': [0, 200], 'y': [1], z: []}
        ```
    clim : List
        [vmin, vmax] for plotting
    cmap : str or Colormap
        colormap, it can be str or matplotlib's Colormap or vispy's Colormap
    show_bar : bool
        show colorbar
    cbar_params : Dict
        parameters pass to colorbar

    type

    Returns
    -------
    traces : List
        List of go.Surface
    """
    
    if cbar_params is None:
        cbar_params={
            'title': 'Min/Max',
            'orientation': 'v'
        }       
    
    cbar_position = kwargs.get('cbar_position')
    if cbar_position is not None:
        if cbar_position=='below':
            cbar_params['orientation']='h'
            cbar_params['y']=-.1
        elif cbar_position=='above':
            cbar_params['orientation']='h'
        elif cbar_position=='left':
            cbar_params['x']=-.1
            
    if type == 'faces' and pos is None:
        x_size, y_size, z_size = volume.shape
        pos=[[0, x_size-1], [0, y_size-1], [0, z_size-1]]

    line_first = cigvis.is_line_first()
    cbar_params['xanchor'] = 'center'
    shape = volume.shape
    nt = shape[2] if line_first else shape[0]

    if pos is None:
        pos = dict(x=[0], y=[0], z=[nt - 1])
    if isinstance(pos, List):
        assert len(pos) == 3
        if isinstance(pos[0], List):
            x, y, z = pos
        else:
            x, y, z = [pos[0]], [pos[1]], [pos[2]]
        pos = {'x': x, 'y': y, 'z': z}
    assert isinstance(pos, Dict)

    if clim is None:
        clim = utils.auto_clim(volume)
    vmin, vmax = clim

    slices, pos = plotlyutils.make_slices(volume, pos=pos)

    dimname = dict(x='inline', y='crossline', z='time')
                      
    if cmap == 'jet-alpha':
        cmap = [[0.0, 'rgba(0,0,131,1)'], 
                [0.2, 'rgba(0,60,170,1)'], 
                [0.4, 'rgba(5,255,255,1)'], 
                [0.5, 'rgba(0,255,0, 0)'], 
                [0.6, 'rgba(255,255,0, 1)'], 
                [0.8, 'rgba(250,0,0, 1)'], 
                [1.0, 'rgba(128,0,0, 1)']]
    elif cmap == 'empty':
        cmap = [[0.0, 'rgba(0,0,131,0)'], 
                [0.5, 'rgba(0,255,0, 0)'], 
                [1.0, 'rgba(128,0,0, 0)']]
    else:
        cmap = colormap.cmap_to_plotly(cmap)
    
    traces = []
  
    idx = 0
    for dim in ['x', 'y', 'z']:
        assert len(slices[dim]) == len(pos[dim])

        for j in range(len(slices[dim])):

            if show_cbar and idx == 0:
                showscale = True
            else:
                showscale = False

            idx += 1

            s = slices[dim][j]
            if scale != 1:
                s = transform.resize(
                    s, (s.shape[0] // scale, s.shape[1] // scale),
                    3,
                    anti_aliasing=True)
            num = pos[dim][j]
            name = f'{dim}/{dimname[dim]}'
            xx, yy, zz = plotlyutils.make_xyz(num, shape, dim, s.shape)
            traces.append(
                go.Surface(x=xx,
                           y=yy,
                           z=zz,
                           surfacecolor=s,
                           colorscale=cmap,
                           cmin=vmin,
                           cmax=vmax,
                           name=name,
                           colorbar=cbar_params,
                           showscale=showscale,
                           showlegend=False)
            )

    return traces

def plot3D(traces,
           aspect: str ='cube',
           show_grid: bool = False,
           title: str = '',
           show_legend: bool = True,
           **kwargs):

    size = kwargs.get('size', (900, 900))
    size = (size, size) if isinstance(size, (int, np.integer)) else size

    scene = kwargs.get('scene', {})

    scened = plotlyutils.make_3Dscene(**kwargs)
    for k, v in scened.items():
        scene.setdefault(k, v)
       
    cols=1
    rows=1
    if isinstance(traces[0], list):
        if len(traces) > 1:
            rows = kwargs.get('rows')
            cols = kwargs.get('cols')
            if cols is None:
                cols=2
            if rows is None:
                if len(traces) % cols > 0:
                    rows = (len(traces) // cols )+1
                else:
                    rows = len(traces) // cols
            spec_item=[]
            spec_i={'is_3d': True}
            for i in range(cols):
                spec_item.append(spec_i)
            specs=[]
            for i in range(rows):
                specs.append(spec_item)
            fig = make_subplots(
                shared_xaxes=True,
                shared_yaxes=True,
                horizontal_spacing=0,
                vertical_spacing=.1,
                rows=rows, 
                cols=cols,
                specs=specs
                )
            for i in range(len(traces)):
                row = (i // cols)+1
                col = i + 1 - ((i//cols)*cols)
                fig.add_traces(data=traces[i], rows=row, cols=col)
        else:
            fig = go.Figure(data=traces[0])
    else:
        fig = go.Figure(data=traces)
    
    #axes
    x_label = kwargs.get('x_label')
    y_label = kwargs.get('y_label')
    z_label = kwargs.get('z_label')

    if x_label is not None:
        scene['xaxis']['title'] = x_label
    if y_label is not None:
        scene['yaxis']['title'] = y_label
    if z_label is not None:
        scene['zaxis']['title'] = z_label

    if show_grid:
        scene['xaxis']['showbackground'] = True
        scene['yaxis']['showbackground'] = True
        scene['zaxis']['showbackground'] = True
    else:
        scene['xaxis']['showbackground'] = False
        scene['yaxis']['showbackground'] = False
        scene['zaxis']['showbackground'] = False

    x_bgcolor = kwargs.get('x_bgcolor')
    y_bgcolor = kwargs.get('y_bgcolor')
    z_bgcolor = kwargs.get('z_bgcolor')
    if x_bgcolor is not None:
        scene['xaxis']['backgroundcolor'] = x_bgcolor
    if y_bgcolor is not None:
        scene['yaxis']['backgroundcolor'] = y_bgcolor
    if z_bgcolor is not None:
        scene['zaxis']['backgroundcolor'] = z_bgcolor

    font_size = kwargs.get('font_size')
    if font_size is None:
        font_size = 12

    x_autorange = kwargs.get('x_autorange')
    if x_autorange is None:
        scene['xaxis']['autorange'] = 'reversed'
    else:
        scene['xaxis']['autorange'] = x_autorange
    y_autorange = kwargs.get('y_autorange')
    if y_autorange is None:
        scene['yaxis']['autorange'] = 'reversed'
    else:
        scene['yaxis']['autorange'] = y_autorange
    z_autorange = kwargs.get('z_autorange')
    if z_autorange is None:
        scene['zaxis']['autorange'] = 'reversed'
    else:
        scene['zaxis']['autorange'] = z_autorange

    x_range = kwargs.get('x_range')
    if x_range is not None:
        scene['xaxis']['range'] = x_range
    y_range = kwargs.get('y_range')
    if y_range is not None:
        scene['yaxis']['range'] = y_range
    z_range = kwargs.get('z_range')
    if z_range is not None:
        scene['zaxis']['range'] = z_range

    x_nticks = kwargs.get('x_nticks')
    if x_nticks is not None:
        scene['xaxis']['nticks'] = x_nticks
    y_nticks = kwargs.get('y_nticks')
    if y_nticks is not None:
        scene['yaxis']['nticks'] = y_nticks
    z_nticks = kwargs.get('z_nticks')
    if z_nticks is not None:
        scene['zaxis']['nticks'] = z_nticks

    x_tickvals = kwargs.get('x_tickvals')
    if x_tickvals is not None:
        scene['xaxis']['tickvals'] = x_tickvals
        scene['xaxis']['ticktext'] = x_tickvals
    y_tickvals = kwargs.get('y_tickvals')
    if y_tickvals is not None:
        scene['yaxis']['tickvals'] = y_tickvals
        scene['yaxis']['ticktext'] = y_tickvals
    z_tickvals = kwargs.get('z_tickvals')
    if z_tickvals is not None:
        scene['zaxis']['tickvals'] = z_tickvals
        scene['zaxis']['ticktext'] = z_tickvals


    annotations = kwargs.get('annotations')
    if annotations is not None:
        scene['annotations'] = annotations            
    camera = kwargs.get('camera')
    if camera is None:
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.25, y=1.25, z=1.25)
        )
    eye = kwargs.get('eye')
    if eye is not None:
        camera['eye'] = eye

    up = kwargs.get('up')
    if up is not None:
        camera['up'] = up

    center = kwargs.get('center')
    if center is not None:
        camera['center'] = center
    
    fig.update_annotations(
            x=30,
            y=1,
            z=4,
            text="Point 2",
            textangle=0,
            ax=0,
            ay=-75,
            font=dict(
                color="black",
                size=12
            ),
            arrowcolor="black",
            arrowsize=3,
            arrowwidth=1,
            arrowhead=1
    )

    fig.update_layout(
        # title=title,
        font_size=font_size,
        # title_text=title,
        height=size[0],
        width=size[1],
        scene1=scene,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=show_legend,
        
        legend={
            "orientation":"h",
            "x": 0.7,
            "yanchor":"top",
            "y": 1,
            "xref": "container",
            "yref": "container",
            "bordercolor":"Black",
            "borderwidth":1,
            "xanchor":"center",
        },
    )

    for i in range(1, (cols * rows)+1):
        exec("fig.update_layout(scene" + str(i) +" = scene)")
        exec("fig.update_layout(scene" + str(i) +"_camera = camera)")

    savequality = kwargs.get('savequality', 1)
    
    fig.update_scenes(
        aspectmode=aspect
    )

    alpha = kwargs.get('alpha')
    if alpha:
        colsc = fig["data"][0]["colorscale"]
        new=[[]]
        flag=False
        for cs in colsc:
            if cs[0] > 0.49 and flag == False:
                new.append([0.5,'rgba(0,255,0, 0)'])
                new.append(cs)
                flag = True
            else:
                new.append(cs)
        new.pop(0)
        

        for f in range(len(fig["data"])):
            fig["data"][f]["colorscale"]  = new

    fig.show(
        config={
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'custom_image',
                'scale': savequality
            }
        })

def create_markers(traces,
                x = [],
                y = [],
                z = [],
                text = "",
                name = "",
                color = "blue",
                group="",
                **kwargs):

    # specify trace names and symbols in a dict
    symbols = {'receiver': 'diamond','source': 'circle'}

    marker_symbol="cross"
    if group in symbols:
        marker_symbol=symbols[group]

    if 'points' in kwargs:
        ponto_marcador = kwargs.get('points')
        if isinstance(ponto_marcador, dict):
            x=ponto_marcador['x']
            y=ponto_marcador['y']
            z=ponto_marcador['z']
        elif len(ponto_marcador) > 0:
            x=[]
            y=[]
            z=[]
            for ponto in ponto_marcador:
                x.append(ponto[0])
                y.append(ponto[1])
                z.append(ponto[2])
        else:
            x=ponto_marcador[0]
            y=ponto_marcador[1]
            z=ponto_marcador[2]

    if 'line_width' in kwargs:
        line_width = kwargs.get('line_width')
    else:
        line_width = 2

    traces.append(
        go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    text=text,
                    mode="markers",
                    marker=dict(
                                size=8,
                                symbol=marker_symbol,
                                # angle=45,
                                line=dict(
                                    color=color,
                                    width=line_width
                                ),

                    ),
                    legendgroup=group,
                    showlegend=True,
                    name=name,
                    hovertemplate="""Y: %{y} <br> X: %{x} <br> Z: %{z} <br> Text: %{text} <br><extra></extra>"""
        )
    )

    return traces

def surface_grid(traces,
                dx, 
                dy,
                **kwargs):

    new_data = traces
    x_size, y_size, z_size = new_data.shape
    max_value = new_data.max()

    xs = x_size // dx
    ys = y_size // dy
    for i in range(1,xs):
        for j in range(y_size):
            new_data[i*dx][j][0]=max_value

    for i in range(1,ys):
        for j in range(x_size):
            new_data[j][i*dy][0]=max_value

    return new_data
        

def surface_intersections(traces,
                dx, 
                dy,
                **kwargs):

    x_size, y_size, z_size = traces.shape
    xs = x_size // dx
    ys = y_size // dy
    pontos=[[]]
    for i in range(1,xs):
        for j in range(1,ys):
            pontos.append([i*dx,j*dy,0])
    pontos.pop(0)            

    return pontos

def piece(traces,
        from_x: int = 0, to_x: int = 0,
        from_y: int = 0, to_y: int = 0,
        from_z: int = 0, to_z: int = 0):

    x_size, y_size, z_size = traces.shape

    if to_x == 0:
        to_x = x_size
    if to_y == 0:
        to_y = y_size
    if to_z == 0:
        to_z = z_size

    new = np.zeros(traces.shape)
    new[from_x:to_x,from_y:to_y,from_z:to_z] = traces[from_x:to_x,from_y:to_y,from_z:to_z]

    return new

def cut(traces,
        from_x: int = 0, to_x: int = 0,
        from_y: int = 0, to_y: int = 0,
        from_z: int = 0, to_z: int = 0):
    
    x_size, y_size, z_size = traces.shape

    if to_x == 0:
        to_x = x_size
    if to_y == 0:
        to_y = y_size
    if to_z == 0:
        to_z = z_size

    empty = np.zeros(traces.shape)
    new = traces

    new[from_x:to_x,from_y:to_y,from_z:to_z] = empty[from_x:to_x,from_y:to_y,from_z:to_z]

    return new
