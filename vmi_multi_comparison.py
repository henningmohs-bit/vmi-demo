"""
VMI (Velocity Map Imaging) Simulation Comparison: Multiple Disturbance Effects
Compares different CAD models against the ideal (normal) reference:
- normal: Ideal model (reference)
- gaps: Gaps in shield
- holes: Holes in lenses
- cables: Cable disturbances
- tiltedshield: Tilted shield (imprecise installation)
- all: All disturbances combined
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import re
import base64
from pathlib import Path

# Publication export dependencies
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import json
import csv
try:
    import pandas as pd
except Exception:
    pd = None

import threading
import sys
import os
from flask import Flask, request, jsonify, send_from_directory, send_file

# Base directory for SIMION models
SIMION_DIR = Path('SIMION')

# Global variables for Flask to serve HTML
GLOBAL_HTML_DIR = '.'
GLOBAL_CURRENT_HTML = 'vmi_multi_comparison.html'

# Configuration for each model - each has a folder with electrode STLs and result file
MODELS = {
    'normal': {
        'name': 'Ideal (Reference)', 
        'color': '#00d4ff',  # Bright cyan - visible in both dark and light mode
        'symbol': 'circle', 
        'is_reference': True, 
        'folder': SIMION_DIR / 'ideal',
        'result_file': 'result',
        'num_electrodes': 9
    },
    'gaps': {
        'name': 'Shield Gaps', 
        'color': 'red', 
        'symbol': 'circle',
        'folder': SIMION_DIR / 'Gaps',
        'result_file': 'result',
        'num_electrodes': 9
    },
    'holes': {
        'name': 'Lens Holes', 
        'color': 'blue', 
        'symbol': 'diamond',
        'folder': SIMION_DIR / 'holes',
        'result_file': 'result',
        'num_electrodes': 9
    },
    'cable': {
        'name': 'Cables', 
        'color': 'green', 
        'symbol': 'square',
        'folder': SIMION_DIR / 'cables',
        'result_file': 'result',
        'num_electrodes': 9
    },
    'tiltedshield': {
        'name': 'Tilted Shield', 
        'color': 'orange', 
        'symbol': 'triangle-up',
        'folder': SIMION_DIR / 'tilted shield',
        'result_file': 'result',
        'num_electrodes': 9
    },
    'all': {
        'name': 'All Disturbances', 
        'color': 'purple', 
        'symbol': 'x',
        'folder': SIMION_DIR / 'all',
        'result_file': 'result',
        'num_electrodes': 9
    },
    # Positive electrodes
    'positive': {
        'name': 'Positive Electrodes (Ideal)',
        'color': 'magenta',
        'symbol': 'diamond',
        'folder': SIMION_DIR / 'positive',
        'result_file': 'result',
        'num_electrodes': 9
    },
    'positive_gaps': {
        'name': 'Positive Electrodes + Gaps',
        'color': '#ff69b4',  # Hot pink
        'symbol': 'diamond-open',
        'folder': SIMION_DIR / 'positive_gaps',
        'result_file': 'result',
        'num_electrodes': 9
    },
    'newshield': {
        'name': 'New Shield Design',
        'color': '#00ff7f',  # Spring green
        'symbol': 'star',
        'folder': SIMION_DIR / 'newshield',
        'result_file': 'result',
        'num_electrodes': 9
    },
}

# ===== UPLOADED MODELS MANAGEMENT =====
UPLOADED_MODELS = {}  # model_key -> {name, color, symbol, data, warnings}
UPLOADED_MODELS_DIR = Path('uploaded_models')
MODEL_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B88B', '#A8DADC']
NEXT_COLOR_INDEX = 0

# Demo datasets (trimmed result files, see prepare_demo_data.py) preloaded on startup
BASE_DIR = Path(__file__).resolve().parent
DEMO_MODEL_FOLDERS = [
    ('Previous design', BASE_DIR / 'demo_data' / 'previous_design'),
    ('Slot-free design', BASE_DIR / 'demo_data' / 'slotfree_design'),
]
# Electrode colors for 3D viewer (distinct colors for each electrode)
ELECTRODE_COLORS = [
    0x3498db,  # Blue - Electrode 1
    0xe74c3c,  # Red - Electrode 2
    0x2ecc71,  # Green - Electrode 3
    0xf39c12,  # Orange - Electrode 4
    0x9b59b6,  # Purple - Electrode 5
    0x1abc9c,  # Teal - Electrode 6
    0xe91e63,  # Pink - Electrode 7
    0x00bcd4,  # Cyan - Electrode 8
    0xcddc39,  # Lime - Electrode 9
]


def parse_trajectory_file(filepath, max_ions=500):
    """
    Parses VMI simulation results with timestep data for trajectory animation.
    Extracts X, Y, Z, TOF, KE per timestep for each ion.
    max_ions: limit number of ions to parse for performance
    """
    trajectories = {}  # ion_n -> list of {tof, x, y, z, ke}
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"    Error reading trajectory file: {e}")
        return trajectories
    
    content = content.replace('\n', ' ')
    ion_blocks = re.split(r'(?=Ion\(\d+\)\s+Event)', content)
    
    for block in ion_blocks:
        if not block.strip():
            continue
        
        ion_match = re.search(r'Ion\((\d+)\)', block)
        if not ion_match:
            continue
        ion_n = int(ion_match.group(1))
        
        if ion_n > max_ions:
            continue
        
        if ion_n not in trajectories:
            trajectories[ion_n] = []
        
        tof_match = re.search(r'TOF\(([\d.e+-]+)\s*usec\)', block)
        x_match = re.search(r'X\(([\d.e+-]+)\s*mm\)', block)
        y_match = re.search(r'Y\(([\d.e+-]+)\s*mm\)', block)
        z_match = re.search(r'Z\(([\d.e+-]+)\s*mm\)', block)
        ke_match = re.search(r'KE\(([\d.e+-]+)\s*eV\)', block)
        
        if x_match and y_match and z_match:
            timestep = {
                'tof': float(tof_match.group(1)) if tof_match else 0.0,
                'x': float(x_match.group(1)),
                'y': float(y_match.group(1)),
                'z': float(z_match.group(1)),
                'ke': float(ke_match.group(1)) if ke_match else 0.0,
            }
            trajectories[ion_n].append(timestep)
    
    return trajectories



# ===== MODEL UPLOAD VALIDATION & PROCESSING =====

def validate_model_upload(model_files):
    """
    Validates uploaded model files and returns status + warnings.
    Expected files: result (required), trajectory_data (optional), *.stl (optional)
    Returns: {valid: bool, has_result: bool, has_trajectory: bool, has_stls: [], warnings: []}
    """
    validation = {
        'valid': False,
        'has_result': False,
        'has_trajectory': False,
        'has_stls': [],
        'warnings': [],
        'errors': []
    }
    
    # Extract just filenames (handle paths like "foldername/result" or "foldername\\result")
    def get_basename(path):
        return path.split('/')[-1].split('\\')[-1].lower()
    
    basenames = [get_basename(f) for f in model_files.keys()]
    
    # Check for result file (REQUIRED)
    if 'result' not in basenames:
        validation['errors'].append('FEHLER: Datei "result" fehlt. Dies ist erforderlich!')
        return validation
    
    validation['has_result'] = True
    validation['valid'] = True
    
    # Check for trajectory data (OPTIONAL)
    # Either a dedicated file OR a large result file (>2MB contains intermediate positions)
    result_content = model_files.get('result', '')
    result_is_large = len(result_content) > 2_000_000
    
    if 'trajectory_data' not in basenames and not result_is_large:
        validation['warnings'].append('WARNING: "trajectory_data" fehlt - Trajektorie nicht verfuegbar')
    else:
        validation['has_trajectory'] = True
    
    # Check for STL files (OPTIONAL) - robustly search by .stl extension
    stl_files = [f for f in model_files.keys() if f.lower().endswith('.stl')]
    if not stl_files:
        validation['warnings'].append('WARNING: Keine STL-Dateien (.stl) gefunden - 3D-Ansicht nicht verfuegbar')
    else:
        validation['has_stls'] = stl_files
    
    return validation


def preload_demo_models():
    """
    Loads the bundled demo datasets (DEMO_MODEL_FOLDERS) into UPLOADED_MODELS at startup,
    so conference visitors see data immediately without needing to upload anything.
    Must be called at module level (not only inside main()) since gunicorn never runs main().
    """
    global UPLOADED_MODELS, NEXT_COLOR_INDEX

    for name, folder in DEMO_MODEL_FOLDERS:
        try:
            result_path = folder / 'result'
            print(f"[PRELOAD] Checking '{name}' at {result_path}", flush=True)
            if not result_path.exists():
                print(f"[PRELOAD] Skipping '{name}': {result_path} not found", flush=True)
                continue

            model_files = {}
            with open(result_path, 'r', encoding='utf-8', errors='ignore') as f:
                model_files['result'] = f.read()

            for stl_path in folder.glob('*.stl'):
                with open(stl_path, 'rb') as f:
                    model_files[stl_path.name.lower()] = f.read()

            validation = validate_model_upload(model_files)
            if validation['errors']:
                print(f"[PRELOAD] '{name}' invalid: {validation['errors']}", flush=True)
                continue

            parsed_data = parse_uploaded_model(name, model_files)

            color = MODEL_COLORS[NEXT_COLOR_INDEX % len(MODEL_COLORS)]
            NEXT_COLOR_INDEX += 1

            model_key = f"demo_{name.replace(' ', '_')}"
            UPLOADED_MODELS[model_key] = {
                'name': name,
                'color': color,
                'symbol': 'circle',
                'data': {'start': parsed_data['start_data'], 'end': parsed_data['end_data']},
                'trajectories': parsed_data['trajectories'],
                'stl_data': parsed_data['stl_data'],
                'has_trajectory': validation['has_trajectory'],
                'has_stls': len(validation['has_stls']) > 0,
                'warnings': parsed_data['warnings']
            }
            print(f"[PRELOAD] Loaded '{name}': {len(parsed_data['start_data']['ion_n'])} ions, "
                  f"{len(parsed_data['stl_data'])} STL files", flush=True)
        except Exception as e:
            import traceback
            print(f"[PRELOAD] FAILED for '{name}': {e}", flush=True)
            traceback.print_exc()


def parse_uploaded_model(model_name, model_files):
    """
    Parse uploaded model files (result + optionals) into the standard data structure.
    Returns: {start_data, end_data, trajectories, stl_data, warnings}
    """
    result = {
        'start_data': {'ion_n': [], 'y': [], 'z': [], 'tof': [], 'ke': []},
        'end_data': {'ion_n': [], 'y': [], 'z': [], 'tof': [], 'ke': []},
        'trajectories': {},
        'stl_data': {},
        'warnings': []
    }
    
    try:
        # Parse result file (REQUIRED)
        if 'result' in model_files:
            try:
                result['start_data'], result['end_data'] = parse_vmi_file_from_content(
                    model_files['result']
                )
            except Exception as e:
                result['warnings'].append(f'? result-Datei konnte nicht geparst werden: {str(e)}')
                return result
        
        # Parse trajectory data (OPTIONAL)
        # First try a dedicated trajectory_data file; if absent, try the result file itself
        # (SIMION sometimes records all timesteps into the result file → large file)
        traj_source = None
        traj_source_name = None
        if 'trajectory_data' in model_files:
            traj_source = model_files['trajectory_data']
            traj_source_name = 'trajectory_data'
        elif 'result' in model_files and len(model_files['result']) > 2_000_000:
            # result file > 2 MB → likely contains intermediate positions
            traj_source = model_files['result']
            traj_source_name = 'result (als Trajektorie)'
        
        if traj_source is not None:
            try:
                result['trajectories'] = parse_trajectory_file_from_content(traj_source)
                print(f"[PARSE] Trajectories from '{traj_source_name}': {len(result['trajectories'])} ions")
            except Exception as e:
                result['warnings'].append(f'trajectory_data konnte nicht geparst werden: {str(e)}')
        
        # Parse STL files (OPTIONAL)
        for stl_name, stl_content in model_files.items():
            if stl_name.lower().endswith('.stl'):
                try:
                    result['stl_data'][stl_name] = base64.b64encode(stl_content).decode('utf-8')
                except Exception as e:
                    result['warnings'].append(f'?? STL-Datei {stl_name} konnte nicht kodiert werden: {str(e)}')
    
    except Exception as e:
        result['warnings'].append(f'? Fehler beim Verarbeiten des Modells: {str(e)}')
    
    return result


def parse_vmi_file_from_content(content_str):
    """Parse VMI result file from string content - copied from working copy4 logic"""
    start_data = {'ion_n': [], 'y': [], 'z': [], 'tof': [], 'ke': []}
    end_data   = {'ion_n': [], 'y': [], 'z': [], 'tof': [], 'ke': []}
    seen_start_ions = set()
    seen_end_ions   = set()

    if isinstance(content_str, bytes):
        content_str = content_str.decode('utf-8', errors='ignore')

    content_str = content_str.replace('\n', ' ')
    ion_blocks  = re.split(r'(?=Ion\(\d+\)\s+Event)', content_str)

    for block in ion_blocks:
        if not block.strip():
            continue
        ion_match = re.search(r'Ion\((\d+)\)', block)
        if not ion_match:
            continue
        ion_n = int(ion_match.group(1))

        event_match = re.search(r'Event\(([^)]+)\)', block)
        if not event_match:
            continue
        event_type = event_match.group(1)

        y_match   = re.search(r'Y\(([\d.e+-]+)\s*mm\)', block)
        z_match   = re.search(r'Z\(([\d.e+-]+)\s*mm\)', block)
        tof_match = re.search(r'TOF\(([\d.e+-]+)\s*usec\)', block)
        ke_match  = re.search(r'KE\(([\d.e+-]+)\s*eV\)', block)

        if not y_match or not z_match:
            continue
        y   = float(y_match.group(1))
        z   = float(z_match.group(1))
        tof = float(tof_match.group(1)) if tof_match else 0.0
        ke  = float(ke_match.group(1))  if ke_match  else 0.0

        if 'Ion Created' in event_type or 'Created' in event_type:
            if ion_n not in seen_start_ions:
                seen_start_ions.add(ion_n)
                start_data['ion_n'].append(ion_n)
                start_data['y'].append(y)
                start_data['z'].append(z)
                start_data['tof'].append(tof)
                start_data['ke'].append(ke)
        elif 'Hit Electrode' in event_type or 'Hit' in event_type:
            if ion_n not in seen_end_ions:
                seen_end_ions.add(ion_n)
                end_data['ion_n'].append(ion_n)
                end_data['y'].append(y)
                end_data['z'].append(z)
                end_data['tof'].append(tof)
                end_data['ke'].append(ke)

    for key in start_data:
        start_data[key] = np.array(start_data[key])
        end_data[key]   = np.array(end_data[key])

    return start_data, end_data


def parse_trajectory_file_from_content(content_str, max_ions=500):
    """Parse trajectory file from string content - copied from working copy4 logic"""
    trajectories = {}

    if isinstance(content_str, bytes):
        content_str = content_str.decode('utf-8', errors='ignore')

    content_str = content_str.replace('\n', ' ')
    ion_blocks  = re.split(r'(?=Ion\(\d+\)\s+Event)', content_str)

    for block in ion_blocks:
        if not block.strip():
            continue
        ion_match = re.search(r'Ion\((\d+)\)', block)
        if not ion_match:
            continue
        ion_n = int(ion_match.group(1))
        if ion_n > max_ions:
            continue
        if ion_n not in trajectories:
            trajectories[ion_n] = []

        tof_match = re.search(r'TOF\(([\d.e+-]+)\s*usec\)', block)
        x_match   = re.search(r'X\(([\d.e+-]+)\s*mm\)', block)
        y_match   = re.search(r'Y\(([\d.e+-]+)\s*mm\)', block)
        z_match   = re.search(r'Z\(([\d.e+-]+)\s*mm\)', block)
        ke_match  = re.search(r'KE\(([\d.e+-]+)\s*eV\)', block)

        if x_match and y_match and z_match:
            timestep = {
                'tof': float(tof_match.group(1)) if tof_match else 0.0,
                'x':   float(x_match.group(1)),
                'y':   float(y_match.group(1)),
                'z':   float(z_match.group(1)),
                'ke':  float(ke_match.group(1)) if ke_match else 0.0,
            }
            trajectories[ion_n].append(timestep)

    return trajectories


def parse_vmi_file(filepath):
    """
    Parses VMI simulation results (SIMION text format) and extracts both start and end positions.
    Also extracts TOF and kinetic energy.
    """
    start_data = {'ion_n': [], 'y': [], 'z': [], 'tof': [], 'ke': []}
    end_data = {'ion_n': [], 'y': [], 'z': [], 'tof': [], 'ke': []}
    
    seen_start_ions = set()
    seen_end_ions = set()
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    content = content.replace('\n', ' ')
    ion_blocks = re.split(r'(?=Ion\(\d+\)\s+Event)', content)
    
    for block in ion_blocks:
        if not block.strip():
            continue
        
        ion_match = re.search(r'Ion\((\d+)\)', block)
        if not ion_match:
            continue
        ion_n = int(ion_match.group(1))
        
        event_match = re.search(r'Event\(([^)]+)\)', block)
        if not event_match:
            continue
        event_type = event_match.group(1)
        
        y_match = re.search(r'Y\(([\d.e+-]+)\s*mm\)', block)
        z_match = re.search(r'Z\(([\d.e+-]+)\s*mm\)', block)
        tof_match = re.search(r'TOF\(([\d.e+-]+)\s*usec\)', block)
        ke_match = re.search(r'KE\(([\d.e+-]+)\s*eV\)', block)
        
        if not y_match or not z_match:
            continue
        y, z = float(y_match.group(1)), float(z_match.group(1))
        tof = float(tof_match.group(1)) if tof_match else 0.0
        ke = float(ke_match.group(1)) if ke_match else 0.0
        
        if 'Ion Created' in event_type or 'Created' in event_type:
            if ion_n not in seen_start_ions:
                seen_start_ions.add(ion_n)
                start_data['ion_n'].append(ion_n)
                start_data['y'].append(y)
                start_data['z'].append(z)
                start_data['tof'].append(tof)
                start_data['ke'].append(ke)
        elif 'Hit Electrode' in event_type or 'Hit' in event_type:
            if ion_n not in seen_end_ions:
                seen_end_ions.add(ion_n)
                end_data['ion_n'].append(ion_n)
                end_data['y'].append(y)
                end_data['z'].append(z)
                end_data['tof'].append(tof)
                end_data['ke'].append(ke)
    
    for key in start_data:
        start_data[key] = np.array(start_data[key])
        end_data[key] = np.array(end_data[key])
    
    return {'start': start_data, 'end': end_data}


def calculate_statistics(data_test, data_ref, pos_type='end'):
    """Calculate deviation statistics between test and reference data."""
    test = data_test[pos_type]
    ref = data_ref[pos_type]
    
    min_len = min(len(test['y']), len(ref['y']))
    if min_len == 0:
        return None
    
    diff_y = test['y'][:min_len] - ref['y'][:min_len]
    diff_z = test['z'][:min_len] - ref['z'][:min_len]
    diff_magnitude = np.sqrt(diff_y**2 + diff_z**2)
    
    return {
        'n_ions': min_len,
        'diff_y': diff_y,
        'diff_z': diff_z,
        'mean_dev': np.mean(diff_magnitude),
        'max_dev': np.max(diff_magnitude),
        'std_dev': np.std(diff_magnitude),
        'dy_min': diff_y.min(), 'dy_max': diff_y.max(),
        'dz_min': diff_z.min(), 'dz_max': diff_z.max(),
    }


CENTER_Y, CENTER_Z = 85.6512, 85.6512


def load_electrode_charges(config):
    """
    Load electrode charges from charges.txt in model folder.
    Returns list of charge values (one per electrode), or empty list if file doesn't exist.
    """
    charges_file = config['folder'] / 'charges.txt'
    charges = []
    if charges_file.exists():
        try:
            with open(charges_file, 'r') as f:
                charges = [line.strip() for line in f.readlines() if line.strip()]
        except Exception as e:
            print(f"    Warning: Could not read charges.txt for {config['name']}: {e}")
    return charges


def create_additional_plots(all_data):
    """Creates additional analysis plots: TOF, Radial, Energy distributions."""
    
    # --- TOF Distribution Plot ---
    tof_fig = make_subplots(rows=1, cols=2,
        subplot_titles=('TOF Distribution', 'TOF Deviation from Ideal [ns]'))
    
    ref_tof = all_data['normal']['end']['tof']
    
    for model_key, model_data in all_data.items():
        config = MODELS[model_key]
        tof = model_data['end']['tof']
        
        tof_fig.add_trace(go.Histogram(
            x=tof, name=config['name'], marker_color=config['color'], opacity=0.6, nbinsx=30,
        ), row=1, col=1)
        
        if model_key != 'normal':
            min_len = min(len(tof), len(ref_tof))
            tof_diff = (tof[:min_len] - ref_tof[:min_len]) * 1000  # Convert to ns
            tof_fig.add_trace(go.Box(
                y=tof_diff, name=config['name'], marker_color=config['color'], boxpoints='outliers',
            ), row=1, col=2)
    
    tof_fig.update_layout(height=400, showlegend=True, barmode='overlay',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(25,25,35,0.8)',
        legend=dict(bgcolor='rgba(30,30,40,0.9)', font=dict(color='#e0e0e0')))
    tof_fig.update_xaxes(title_text='Time of Flight [&micro;s]', row=1, col=1, gridcolor='rgba(100,100,120,0.3)')
    tof_fig.update_yaxes(title_text='Count', row=1, col=1, gridcolor='rgba(100,100,120,0.3)')
    tof_fig.update_yaxes(title_text='dTOF [ns]', row=1, col=2, gridcolor='rgba(100,100,120,0.3)')
    
    # --- Radial Distribution Plot ---
    radial_fig = make_subplots(rows=1, cols=2,
        subplot_titles=('Radial Distribution from Center', 'Radial Deviation [mm]'))
    
    ref_data = all_data['normal']['end']
    ref_radius = np.sqrt((ref_data['y'] - CENTER_Y)**2 + (ref_data['z'] - CENTER_Z)**2)
    
    for model_key, model_data in all_data.items():
        config = MODELS[model_key]
        end = model_data['end']
        radius = np.sqrt((end['y'] - CENTER_Y)**2 + (end['z'] - CENTER_Z)**2)
        
        radial_fig.add_trace(go.Scatter(
            x=end['ion_n'], y=radius, mode='markers',
            marker=dict(size=5, color=config['color'], opacity=0.6), name=config['name'],
        ), row=1, col=1)
        
        if model_key != 'normal':
            min_len = min(len(radius), len(ref_radius))
            radius_diff = radius[:min_len] - ref_radius[:min_len]
            radial_fig.add_trace(go.Histogram(
                x=radius_diff, name=config['name'], marker_color=config['color'],
                opacity=0.6, nbinsx=25, showlegend=False,
            ), row=1, col=2)
    
    radial_fig.update_layout(height=400, showlegend=True, barmode='overlay',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(25,25,35,0.8)',
        legend=dict(bgcolor='rgba(30,30,40,0.9)', font=dict(color='#e0e0e0')))
    radial_fig.update_xaxes(title_text='Ion Number', row=1, col=1, gridcolor='rgba(100,100,120,0.3)')
    radial_fig.update_yaxes(title_text='Radius [mm]', row=1, col=1, gridcolor='rgba(100,100,120,0.3)')
    radial_fig.update_xaxes(title_text='Radial Deviation [mm]', row=1, col=2, gridcolor='rgba(100,100,120,0.3)')
    
    # --- Energy Distribution Plot ---
    energy_fig = make_subplots(rows=1, cols=2,
        subplot_titles=('Kinetic Energy Distribution', 'Energy Deviation [eV]'))
    
    ref_ke = all_data['normal']['end']['ke']
    
    for model_key, model_data in all_data.items():
        config = MODELS[model_key]
        ke = model_data['end']['ke']
        
        energy_fig.add_trace(go.Histogram(
            x=ke, name=config['name'], marker_color=config['color'], opacity=0.6, nbinsx=30,
        ), row=1, col=1)
        
        if model_key != 'normal':
            min_len = min(len(ke), len(ref_ke))
            ke_diff = ke[:min_len] - ref_ke[:min_len]
            energy_fig.add_trace(go.Box(
                y=ke_diff, name=config['name'], marker_color=config['color'], boxpoints='outliers',
            ), row=1, col=2)
    
    energy_fig.update_layout(height=400, showlegend=True, barmode='overlay',
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(25,25,35,0.8)',
        legend=dict(bgcolor='rgba(30,30,40,0.9)', font=dict(color='#e0e0e0')))
    energy_fig.update_xaxes(title_text='Kinetic Energy [eV]', row=1, col=1, gridcolor='rgba(100,100,120,0.3)')
    energy_fig.update_yaxes(title_text='Count', row=1, col=1, gridcolor='rgba(100,100,120,0.3)')
    energy_fig.update_yaxes(title_text='dKE [eV]', row=1, col=2, gridcolor='rgba(100,100,120,0.3)')
    
    return tof_fig, radial_fig, energy_fig


# -------------------- Publication-ready export helpers --------------------
def compute_centroid(y_arr, z_arr):
    """Return centroid (y,z), radial shift and spread metrics."""
    if len(y_arr) == 0:
        return None
    yc = float(np.mean(y_arr))
    zc = float(np.mean(z_arr))
    dy = y_arr - yc
    dz = z_arr - zc
    rms = float(np.sqrt(np.mean(dy**2 + dz**2)))
    std_y = float(np.std(y_arr))
    std_z = float(np.std(z_arr))
    return {"yc": yc, "zc": zc, "rms": rms, "std_y": std_y, "std_z": std_z}


def plot_detector_2d_matplotlib(all_data, reference_key, model_key,
                                out_prefix='detector', formats=('png','svg','pdf'), dpi=300,
                                show_ellipses=True, save_stats=True):
    """Create a publication-quality 2D detector plot (Y vs Z) using matplotlib.

    - all_data: parsed simulation data (same structure used elsewhere)
    - reference_key: key for the ideal/reference model (e.g., 'normal')
    - model_key: key for the modified model to compare
    - out_prefix: output filename prefix
    - formats: tuple of formats to write (png/svg/pdf)
    - dpi: dpi for raster exports
    - show_ellipses: draw 1-sigma ellipses for both datasets
    - save_stats: also save CSV/JSON with calculated metrics
    """
    ref = all_data[reference_key]['end']
    test = all_data[model_key]['end']

    y_ref, z_ref = np.array(ref['y']), np.array(ref['z'])
    y_test, z_test = np.array(test['y']), np.array(test['z'])

    # Compute centroids and statistics
    c_ref = compute_centroid(y_ref, z_ref)
    c_test = compute_centroid(y_test, z_test)

    if c_ref is None or c_test is None:
        print('Not enough data to produce detector plot for', model_key)
        return None

    dy = c_test['yc'] - c_ref['yc']
    dz = c_test['zc'] - c_ref['zc']
    radial_shift = float(np.sqrt(dy**2 + dz**2))

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.set_aspect('equal')

    # Scatter reference and test impacts
    ax.scatter(y_ref, z_ref, s=25, facecolors='none', edgecolors=MODELS[reference_key]['color'], label=MODELS[reference_key]['name'], alpha=0.9)
    ax.scatter(y_test, z_test, s=18, c=MODELS[model_key]['color'], label=MODELS[model_key]['name'], alpha=0.8)

    # Centroids
    ax.plot(c_ref['yc'], c_ref['zc'], marker='D', markersize=10, color=MODELS[reference_key]['color'], label='Centroid (Ideal)')
    ax.plot(c_test['yc'], c_test['zc'], marker='*', markersize=12, color=MODELS[model_key]['color'], label='Centroid (Modified)')

    # Dashed reference lines at ideal centroid
    ax.axvline(c_ref['yc'], color='gray', linestyle='--', linewidth=1)
    ax.axhline(c_ref['zc'], color='gray', linestyle='--', linewidth=1)

    # Arrow showing centroid shift
    ax.annotate('', xy=(c_test['yc'], c_test['zc']), xytext=(c_ref['yc'], c_ref['zc']),
                arrowprops=dict(arrowstyle='->', color='yellow', lw=2))

    # Optional ellipses for 1-sigma
    if show_ellipses:
        def add_sigma_ellipse(ax, y_arr, z_arr, center, edgecolor, facecolor=None):
            cov = np.cov(y_arr, z_arr)
            if np.any(np.isnan(cov)):
                return
            vals, vecs = np.linalg.eigh(cov)
            if np.any(vals <= 0):
                return
            # 1-sigma ellipse (chi2 ~ 2.295 for 68% in 2D is 2.296, but use sqrt(vals) scaling)
            width, height = 2 * np.sqrt(vals)
            angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
            ell = Ellipse((center[0], center[1]), width=width, height=height,
                          angle=angle, edgecolor=edgecolor, facecolor=facecolor, alpha=0.25, lw=1.5)
            ax.add_patch(ell)

        add_sigma_ellipse(ax, y_ref, z_ref, (c_ref['yc'], c_ref['zc']), edgecolor=MODELS[reference_key]['color'], facecolor=MODELS[reference_key]['color'])
        add_sigma_ellipse(ax, y_test, z_test, (c_test['yc'], c_test['zc']), edgecolor=MODELS[model_key]['color'], facecolor=MODELS[model_key]['color'])

    # Labels and annotation
    ax.set_xlabel('Y Position [mm]')
    ax.set_ylabel('Z Position [mm]')
    ax.set_title(f"Detector Impacts: {MODELS[reference_key]['name']} vs {MODELS[model_key]['name']}")
    ax.legend(loc='upper right', fontsize='small')

    # Annotation about shift
    ax.text(0.02, 0.02, f"Centroid shift: {radial_shift:.3f} mm (&Delta;Y={dy:.3f}, &Delta;Z={dz:.3f})\nStd (ref): y={c_ref['std_y']:.3f}, z={c_ref['std_z']:.3f}",
            transform=ax.transAxes, fontsize=9, color='white', bbox=dict(facecolor='black', alpha=0.4))

    # Tight layout and save
    fig.tight_layout()
    for fmt in formats:
        out_file = f"{out_prefix}_{model_key}.{fmt}"
        try:
            if fmt.lower() == 'png':
                fig.savefig(out_file, dpi=dpi, bbox_inches='tight')
            else:
                fig.savefig(out_file, bbox_inches='tight')
            print('Saved', out_file)
        except Exception as e:
            print('Could not save', out_file, '->', e)

    # Optional quantitative export
    if save_stats:
        stats = {
            'model': model_key,
            'n_ref': int(len(y_ref)),
            'n_test': int(len(y_test)),
            'centroid_ref_y': c_ref['yc'], 'centroid_ref_z': c_ref['zc'],
            'centroid_test_y': c_test['yc'], 'centroid_test_z': c_test['zc'],
            'delta_y': dy, 'delta_z': dz, 'radial_shift': radial_shift,
            'rms_ref': c_ref['rms'], 'rms_test': c_test['rms'],
            'std_y_ref': c_ref['std_y'], 'std_z_ref': c_ref['std_z'],
            'std_y_test': c_test['std_y'], 'std_z_test': c_test['std_z']
        }
        csv_file = f"{out_prefix}_{model_key}.csv"
        json_file = f"{out_prefix}_{model_key}.json"
        try:
            with open(csv_file, 'w', newline='') as cf:
                writer = csv.writer(cf)
                for k, v in stats.items():
                    writer.writerow([k, v])
            with open(json_file, 'w') as jf:
                json.dump(stats, jf, indent=2)
            print('Saved stats', csv_file, json_file)
        except Exception as e:
            print('Could not save stats ->', e)

    plt.close(fig)
    return {'plot_file_prefix': out_prefix, 'stats': stats}


def export_3d_plotly_geometry_and_trajectories(model_key, all_data, trajectory_data,
                                               out_prefix='3d_geometry', formats=('png','svg'), dpi=300,
                                               max_trajs=200):
    """Create a Plotly 3D figure of geometry and trajectories and export to vector/raster formats.

    Improved STL parsing (ASCII or binary) and trajectories colored by kinetic energy (heatmap).
    Requires kaleido for fig.write_image to work.
    """
    config = MODELS.get(model_key)
    if config is None:
        print('Unknown model for 3D export:', model_key)
        return None

    fig = go.Figure()

    folder = config.get('folder')

    def parse_stl(path):
        """Return (verts, i,j,k) arrays for triangles. Tries ASCII then binary STL."""
        verts = []
        faces_i = []
        faces_j = []
        faces_k = []
        try:
            with open(path, 'rb') as f:
                data = f.read()
            text = None
            try:
                text = data.decode('utf-8')
            except Exception:
                text = None
            if text and 'vertex' in text:
                # ASCII parse
                vlist = []
                for line in text.splitlines():
                    line = line.strip()
                    if line.lower().startswith('vertex'):
                        parts = line.split()
                        if len(parts) >= 4:
                            vlist.append([float(parts[1]), float(parts[2]), float(parts[3])])
                if vlist:
                    # Build triangles in sequence
                    verts = []
                    idx = 0
                    for t in range(0, len(vlist), 3):
                        if t+2 < len(vlist):
                            v0, v1, v2 = vlist[t], vlist[t+1], vlist[t+2]
                            verts.extend([v0, v1, v2])
                            faces_i.append(idx); faces_j.append(idx+1); faces_k.append(idx+2)
                            idx += 3
                    return np.array(verts), np.array(faces_i,dtype=int), np.array(faces_j,dtype=int), np.array(faces_k,dtype=int)
            # Binary STL parse
            import struct
            header = data[:80]
            if len(data) < 84:
                return None, None, None, None
            ntri = struct.unpack('<I', data[80:84])[0]
            offset = 84
            idx = 0
            for t in range(ntri):
                if offset + 50 > len(data):
                    break
                # normal + 3 vertices + attr
                vals = struct.unpack('<12fH', data[offset:offset+50])
                vx1 = [vals[3], vals[4], vals[5]]
                vx2 = [vals[6], vals[7], vals[8]]
                vx3 = [vals[9], vals[10], vals[11]]
                verts.extend([vx1, vx2, vx3])
                faces_i.append(idx); faces_j.append(idx+1); faces_k.append(idx+2)
                idx += 3
                offset += 50
            if verts:
                return np.array(verts), np.array(faces_i,dtype=int), np.array(faces_j,dtype=int), np.array(faces_k,dtype=int)
        except Exception as e:
            return None, None, None, None
        return None, None, None, None

    # Add electrode mesh surfaces if STL exists
    if folder and folder.exists():
        for i in range(1, config.get('num_electrodes', 9) + 1):
            stl_file = folder / f'electrode{i}.stl'
            if stl_file.exists():
                verts, fi, fj, fk = parse_stl(str(stl_file))
                if verts is not None and fi is not None and len(verts):
                    fig.add_trace(go.Mesh3d(
                        x=verts[:,0], y=verts[:,1], z=verts[:,2],
                        i=fi, j=fj, k=fk,
                        opacity=0.9,
                        color='lightgrey',
                        name=f'Electrode {i}',
                        flatshading=True,
                        showscale=False
                    ))

    # Plot ionization/sample volume as small translucent sphere (approx)
    # create sphere mesh
    u = np.linspace(0, 2*np.pi, 20)
    v = np.linspace(0, np.pi, 10)
    r = 2.0
    xs = (r * np.outer(np.cos(u), np.sin(v))).flatten()
    ys = (r * np.outer(np.sin(u), np.sin(v))).flatten()
    zs = (r * np.outer(np.ones_like(u), np.cos(v))).flatten()
    fig.add_trace(go.Mesh3d(x=xs, y=ys, z=zs, opacity=0.25, color='white', name='Ionization volume'))

    # Trajectories colored by KE
    trajs = trajectory_data.get(model_key, {})
    max_ke = 0.0
    for steps in trajs.values():
        kes = [s.get('ke', 0.0) for s in steps]
        if len(kes):
            max_ke = max(max_ke, max(kes))
    if max_ke <= 0:
        max_ke = None

    count = 0
    colorbar_added = False
    for ion_n, steps in trajs.items():
        if count >= max_trajs:
            break
        xs = np.array([s['x'] for s in steps])
        ys = np.array([s['y'] for s in steps])
        zs = np.array([s['z'] for s in steps])
        kes = np.array([s.get('ke', 0.0) for s in steps])
        if len(xs) < 2:
            continue
        line_kwargs = dict(width=2)
        if kes is not None and len(kes):
            line_kwargs['color'] = kes
            line_kwargs['colorscale'] = 'Viridis'
            # only show colorbar once
            if not colorbar_added:
                line_kwargs['showscale'] = True
                line_kwargs['colorbar'] = dict(title='KE [eV]')
                colorbar_added = True
            else:
                line_kwargs['showscale'] = False
        else:
            line_kwargs['color'] = MODELS[model_key]['color']
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode='lines',
                                   line=line_kwargs,
                                   name=f'Traj {ion_n}', showlegend=False))
        count += 1

    # Add end impact detector points
    end = all_data[model_key]['end']
    if len(end['y']):
        det_y = np.mean(end['y'])
        det_z = np.mean(end['z'])
        det_x = 0
        fig.add_trace(go.Scatter3d(x=[det_x], y=[det_y], z=[det_z], mode='markers+text', marker=dict(size=6, color='gold'), text=['Detector'], textposition='top center', name='Detector'))

    # Annotations (plotly text)
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='text', text=['0 V region &rarr; increasingly negative potentials'], showlegend=False))

    fig.update_layout(scene=dict(xaxis_title='X [mm]', yaxis_title='Y [mm]', zaxis_title='Z [mm]', bgcolor='white'),
                      title=f"3D Geometry & Trajectories: {config['name']}",
                      paper_bgcolor='white')

    # Export images via kaleido
    for fmt in formats:
        out_file = f"{out_prefix}_{model_key}.{fmt}"
        try:
            if fmt.lower() == 'png':
                fig.write_image(out_file, width=int(8*dpi), height=int(6*dpi), scale=1)
            else:
                fig.write_image(out_file)
            print('Saved 3D export', out_file)
        except Exception as e:
            print('Could not write 3D export', out_file, '->', e)


    return fig


def create_multi_comparison_plot(all_data, reference_key='normal', ions_loaded=None):
    """
    Creates a comparison plot with all models overlaid and difference plots.
    Includes dropdown to switch between start and end positions.
    ions_loaded: dict with model_key -> number of ions loaded
    """
    ref_data = all_data[reference_key]
    test_models = {k: v for k, v in all_data.items() if k != reference_key}
    num_test_models = len(test_models)
    
    # Create subplots: 1x2 layout
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            'Overlay: All Models vs <b>Ideal Reference</b>',
            'Position Deviations from Ideal (&Delta;Y, &Delta;Z)'
        ),
        specs=[[{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    all_stats = {}
    
    # Create traces for both position types (end and start)
    for pos_type in ['end', 'start']:
        is_visible = (pos_type == 'end')  # End positions visible by default
        
        # Add center lines for overlay plot
        fig.add_trace(go.Scatter(
            x=[75, 95], y=[85.6512, 85.6512], mode='lines',
            line=dict(color='gray', width=1, dash='dash'),
            showlegend=False, hoverinfo='skip', visible=is_visible
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[85.6512, 85.6512], y=[75, 95], mode='lines',
            line=dict(color='gray', width=1, dash='dash'),
            showlegend=False, hoverinfo='skip', visible=is_visible
        ), row=1, col=1)
        
        # Add center lines for difference plot
        fig.add_trace(go.Scatter(
            x=[-3, 3], y=[0, 0], mode='lines',
            line=dict(color='gray', width=1, dash='dash'),
            showlegend=False, hoverinfo='skip', visible=is_visible
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=[0, 0], y=[-3, 3], mode='lines',
            line=dict(color='gray', width=1, dash='dash'),
            showlegend=False, hoverinfo='skip', visible=is_visible
        ), row=1, col=2)
        
        # Plot reference (ideal) model
        ref_config = MODELS[reference_key]
        ref_pos = ref_data[pos_type]
        fig.add_trace(go.Scatter(
            x=ref_pos['y'], y=ref_pos['z'],
            mode='markers',
            marker=dict(size=12, color=ref_config['color'], opacity=0.8, symbol='circle-open', line=dict(width=2)),
            name=ref_config['name'],
            hovertemplate=f"<b>{ref_config['name']}</b><br>Ion: %{{text}}<br>Y: %{{x:.2f}} mm<br>Z: %{{y:.2f}} mm<extra></extra>",
            text=ref_pos['ion_n'],
            legendgroup='ref',
            visible=is_visible
        ), row=1, col=1)
        
        # Plot each test model
        for model_key, model_data in test_models.items():
            config = MODELS[model_key]
            pos_data = model_data[pos_type]
            
            # Overlay plot
            fig.add_trace(go.Scatter(
                x=pos_data['y'], y=pos_data['z'],
                mode='markers',
                marker=dict(size=8, color=config['color'], opacity=0.6, symbol=config['symbol']),
                name=config['name'],
                hovertemplate=f"<b>{config['name']}</b><br>Ion: %{{text}}<br>Y: %{{x:.2f}} mm<br>Z: %{{y:.2f}} mm<extra></extra>",
                text=pos_data['ion_n'],
                legendgroup=model_key,
                visible=is_visible
            ), row=1, col=1)
            
            # Calculate and plot differences
            stats = calculate_statistics(model_data, ref_data, pos_type)
            if stats:
                if pos_type == 'end':
                    all_stats[model_key] = stats
                fig.add_trace(go.Scatter(
                    x=stats['diff_y'], y=stats['diff_z'],
                    mode='markers',
                    marker=dict(size=8, color=config['color'], opacity=0.7, symbol=config['symbol']),
                    name=f"&Delta; {config['name']}",
                    hovertemplate=f"<b>{config['name']}</b><br>Ion: %{{text}}<br>&Delta;Y: %{{x:.4f}} mm<br>&Delta;Z: %{{y:.4f}} mm<extra></extra>",
                    text=ref_pos['ion_n'][:stats['n_ions']],
                    legendgroup=model_key,
                    showlegend=False,
                    visible=is_visible
                ), row=1, col=2)
    
    # Calculate number of traces per position type
    # 4 center lines + 1 reference + num_test_models overlay + num_test_models difference = 4 + 1 + 2*num_test_models
    num_traces_per_type = 4 + 1 + 2 * num_test_models
    
    # Create dropdown menu buttons
    buttons = [
        dict(
            label="End Positions (Detector)",
            method="update",
            args=[
                {"visible": [True] * num_traces_per_type + [False] * num_traces_per_type},
                {"title": "VMI Simulation: Disturbance Effects &mdash; End Positions (Detector)"}
            ]
        ),
        dict(
            label="Start Positions (Source)",
            method="update",
            args=[
                {"visible": [False] * num_traces_per_type + [True] * num_traces_per_type},
                {"title": "VMI Simulation: Disturbance Effects &mdash; Start Positions (Source)"}
            ]
        )
    ]
    
    # Build statistics text with ions loaded info
    if ions_loaded is None:
        ions_loaded = {}
    
    stats_lines = ["<b>Ions Loaded per Model</b>"]
    for model_key in all_data.keys():
        config = MODELS[model_key]
        n_ions = ions_loaded.get(model_key, len(all_data[model_key]['end']['y']))
        stats_lines.append(
            f"<span style='color:{config['color']}'>{config['name']}</span>: {n_ions} ions"
        )
    
    stats_lines.append("")
    stats_lines.append("<b>Deviation Statistics - End Positions (vs Ideal)</b>")
    for model_key, stats in all_stats.items():
        config = MODELS[model_key]
        stats_lines.append(
            f"<span style='color:{config['color']}'><b>{config['name']}</b></span>: "
            f"Mean={stats['mean_dev']:.3f}, Max={stats['max_dev']:.3f}, Std={stats['std_dev']:.3f} mm"
        )
    
    # Update layout with dark theme colors (no dropdown - using HTML tabs instead)
    fig.update_layout(
        title=dict(
            text="",  # Title will be in HTML header
            font=dict(size=18, color='#e0e0e0'),
            y=0.98, x=0.5, xanchor='center'
        ),
        showlegend=True,
        legend=dict(
            x=1.02, y=0.5,
            bgcolor='rgba(30,30,40,0.9)',
            bordercolor='#444', borderwidth=1,
            font=dict(color='#e0e0e0')
        ),
        height=550,
        width=1100,
        margin=dict(t=60, b=60, l=60, r=220),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(25,25,35,0.8)'
    )
    
    # Axis settings with dark theme
    axis_style = dict(
        gridcolor='rgba(100,100,120,0.3)',
        linecolor='#444',
        tickfont=dict(color='#aaa'),
        title_font=dict(color='#ccc')
    )
    fig.update_xaxes(title_text='Y Position [mm]', scaleanchor='y', scaleratio=1, row=1, col=1, **axis_style)
    fig.update_yaxes(title_text='Z Position [mm]', row=1, col=1, **axis_style)
    fig.update_xaxes(title_text='&Delta;Y [mm]', scaleanchor='y2', scaleratio=1, row=1, col=2, **axis_style)
    fig.update_yaxes(title_text='&Delta;Z [mm]', row=1, col=2, **axis_style)
    fig.update_yaxes(title_text='&Delta;Z [mm]', row=1, col=2)
    
    return fig, all_stats


def main():
    print("=" * 60)
    print("VMI Simulation Comparison: Multiple Disturbance Effects")
    print("=" * 60)
    print("\nWaiting for model upload...")
    print("   Use the upload interface to load your simulation models.")
    print("   Required: result file")
    print("   Optional: trajectory_data, electrode STLs\n")
    
    # Initialize empty data - models will be loaded via upload interface
    all_data = {}
    trajectory_data = {}
    
    # Expose empty data structure to Flask export server
    global GLOBAL_ALL_DATA, GLOBAL_TRAJ_DATA
    GLOBAL_ALL_DATA = all_data
    GLOBAL_TRAJ_DATA = trajectory_data
    
    print('-' * 60)
    print("All values in mm")
    print('=' * 60)
    
    # Save with custom HTML including checkboxes
    output_file = 'vmi_multi_comparison.html'
    
    # Build statistics for display (will be populated after models are uploaded)
    stats_html = ""
    
    # Modern Dark Mode page wrapper with Light/Dark toggle
    page_header = '''<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VMI Simulation Analysis</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        :root {
            --bg-primary: linear-gradient(135deg, #0d0d1a 0%, #1a1a2e 50%, #16213e 100%);
            --bg-card: rgba(30, 30, 45, 0.8);
            --bg-card-hover: rgba(255,255,255,0.08);
            --bg-input: rgba(255,255,255,0.03);
            --border-color: rgba(255,255,255,0.08);
            --text-primary: #e0e0e0;
            --text-secondary: #888;
            --text-muted: #aaa;
            --plot-bg: rgba(25,25,35,0.8);
            --stl-bg: #15151f;
        }
        
        [data-theme="light"] {
            --bg-primary: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 50%, #d9e2ec 100%);
            --bg-card: rgba(255, 255, 255, 0.9);
            --bg-card-hover: rgba(0,0,0,0.05);
            --bg-input: rgba(0,0,0,0.03);
            --border-color: rgba(0,0,0,0.1);
            --text-primary: #1a1a2e;
            --text-secondary: #555;
            --text-muted: #666;
            --plot-bg: rgba(250,250,255,0.9);
            --stl-bg: #f0f0f5;
        }
        
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: var(--bg-primary);
            min-height: 100vh;
            color: var(--text-primary);
            transition: all 0.3s ease;
        }
        .container {
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
        }
        header {
            text-align: center;
            padding: 30px 0 20px;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 25px;
            position: relative;
        }
        header h1 {
            font-size: 2.2em;
            font-weight: 300;
            background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 8px;
        }
        header p {
            color: var(--text-secondary);
            font-size: 1.1em;
        }
        
        /* Theme Toggle Switch */
        .theme-toggle {
            position: absolute;
            top: 50%;
            right: 20px;
            transform: translateY(-50%);
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .theme-toggle span {
            font-size: 1.2em;
        }
        .toggle-switch {
            position: relative;
            width: 60px;
            height: 30px;
        }
        .toggle-switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }
        .toggle-slider {
            position: absolute;
            cursor: pointer;
            top: 0; left: 0; right: 0; bottom: 0;
            background: linear-gradient(135deg, #667eea, #764ba2);
            border-radius: 30px;
            transition: 0.3s;
        }
        .toggle-slider:before {
            position: absolute;
            content: "";
            height: 24px;
            width: 24px;
            left: 3px;
            bottom: 3px;
            background: white;
            border-radius: 50%;
            transition: 0.3s;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }
        .toggle-switch input:checked + .toggle-slider {
            background: linear-gradient(135deg, #f093fb, #f5576c);
        }
        .toggle-switch input:checked + .toggle-slider:before {
            transform: translateX(30px);
        }
        
        .main-grid {
            display: grid;
            grid-template-columns: 340px 1fr;
            gap: 25px;
            align-items: start;
        }
        .sidebar {
            background: var(--bg-card);
            border-radius: 16px;
            padding: 20px;
            border: 1px solid var(--border-color);
            backdrop-filter: blur(10px);
            position: sticky;
            top: 20px;
        }
        .sidebar h3 {
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-secondary);
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid var(--border-color);
        }
        .model-item {
            display: flex;
            align-items: center;
            padding: 10px 12px;
            margin: 6px 0;
            border-radius: 10px;
            background: var(--bg-input);
            transition: all 0.2s ease;
        }
        .model-item:hover {
            background: var(--bg-card-hover);
        }
        .model-item input[type="checkbox"] {
            width: 18px;
            height: 18px;
            margin-right: 12px;
            accent-color: #667eea;
            cursor: pointer;
        }
        .model-item label {
            flex: 1;
            font-weight: 500;
            cursor: pointer;
        }
        .model-item .btn-3d {
            padding: 4px 10px;
            font-size: 11px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            border: none;
            border-radius: 6px;
            color: white;
            cursor: pointer;
            transition: all 0.2s ease;
            font-weight: 500;
        }
        .model-item .btn-3d:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        .btn-update {
            width: 100%;
            padding: 14px;
            margin-top: 20px;
            font-size: 14px;
            font-weight: 600;
            background: linear-gradient(135deg, #11998e, #38ef7d);
            border: none;
            border-radius: 10px;
            color: white;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .btn-update:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(56, 239, 125, 0.3);
        }
        .content-area {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .plot-container {
            background: var(--bg-card);
            border-radius: 16px;
            padding: 20px;
            border: 1px solid var(--border-color);
        }
        .position-plots-row {
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
            gap: 18px;
            align-items: start;
        }
        .position-plots-row.single {
            grid-template-columns: minmax(0, 1fr);
        }
        .position-plot-cell {
            min-width: 0;
        }
        #plotly-container,
        #deviation-container {
            width: 100%;
            height: 480px;
        }
        @media (max-width: 1200px) {
            .position-plots-row {
                grid-template-columns: minmax(0, 1fr);
            }
        }
        .plot-title {
            font-size: 1.1em;
            font-weight: 500;
            margin-bottom: 15px;
            color: var(--text-muted);
        }
        .stats-panel {
            background: var(--bg-card);
            border-radius: 16px;
            padding: 20px;
            border: 1px solid var(--border-color);
        }
        .stats-panel h3 {
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-secondary);
            margin-bottom: 15px;
        }
        .stat-row {
            display: flex;
            justify-content: space-between;
            padding: 10px 15px;
            margin: 5px 0;
            background: var(--bg-input);
            border-radius: 8px;
        }
        .stat-name {
            font-weight: 600;
            min-width: 140px;
        }
        .stat-value {
            font-family: 'SF Mono', 'Monaco', monospace;
            font-size: 0.9em;
            color: var(--text-muted);
        }
        .stl-viewer-section {
            background: var(--bg-card);
            border-radius: 16px;
            padding: 20px;
            border: 1px solid var(--border-color);
            display: none;
            overflow: hidden;
        }
        .stl-viewer-section.active {
            display: block;
        }
        .stl-viewer-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        .stl-viewer-header h3 {
            font-size: 1.1em;
            font-weight: 500;
            color: var(--text-muted);
        }
        .btn-close-3d {
            padding: 8px 16px;
            background: rgba(244, 67, 54, 0.8);
            border: none;
            border-radius: 8px;
            color: white;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        .btn-close-3d:hover {
            background: #f44336;
        }
        #stlContainer {
            width: 100%;
            height: 420px;
            border-radius: 10px;
            overflow: hidden;
            background: var(--stl-bg);
        }
        #trajControls {
            display: none;
        }
        .stl-viewer-section.active #trajControls {
            display: block;
        }
        .js-plotly-plot .plotly .modebar {
            background: var(--bg-card) !important;
        }
        .js-plotly-plot .plotly .modebar-btn path {
            fill: var(--text-secondary) !important;
        }
        
        /* Tab Bar Styles */
        .tab-bar {
            display: flex;
            gap: 0;
            margin-bottom: 15px;
            background: var(--bg-input);
            border-radius: 12px;
            padding: 4px;
            border: 1px solid var(--border-color);
        }
        .tab-btn {
            flex: 1;
            padding: 12px 20px;
            border: none;
            background: transparent;
            color: var(--text-secondary);
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            border-radius: 8px;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }
        .tab-btn:hover {
            background: var(--bg-card-hover);
            color: var(--text-primary);
        }
        .tab-btn.active {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        .tab-btn .tab-icon {
            font-size: 16px;
        }
        
        /* Main Tab Navigation */
        .main-tab-bar {
            display: flex;
            gap: 0;
            margin-bottom: 20px;
            background: var(--bg-input);
            border-radius: 12px;
            padding: 4px;
            border: 1px solid var(--border-color);
        }
        .main-tab-btn {
            flex: 1;
            padding: 14px 16px;
            border: none;
            background: transparent;
            color: var(--text-secondary);
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            border-radius: 8px;
            transition: all 0.3s ease;
            text-align: center;
        }
        .main-tab-btn:hover {
            background: var(--bg-card-hover);
            color: var(--text-primary);
        }
        .main-tab-btn.active {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        .main-tab-panel {
            /* Initially visible for Plotly rendering */
        }
        .main-tab-panel.hidden {
            display: none;
        }
        .main-tab-panel.active {
            display: block !important;
        }
        
        /* Brainrot Mode Styles */
        .brainrot-toggle {
            position: absolute;
            top: 50%;
            left: 20px;
            transform: translateY(-50%);
            display: flex;
            align-items: center;
            gap: 8px;
        }

        /* ===== Mobile / Tablet Responsiveness ===== */
        @media (max-width: 900px) {
            .main-grid {
                grid-template-columns: 1fr;
            }
            .sidebar {
                position: static;
            }
        }
        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }
            header {
                padding: 70px 0 15px;
            }
            header h1 {
                font-size: 1.5em;
            }
            header p {
                font-size: 0.9em;
            }
            .theme-toggle {
                top: 12px;
                right: 10px;
                transform: none;
            }
            .header-toolbar {
                justify-content: center;
            }
            .header-toolbar .tab-btn {
                font-size: 12px;
                padding: 10px 12px;
                flex: 1 1 auto;
            }
            .main-grid {
                gap: 15px;
            }
            .sidebar {
                padding: 14px;
            }
            .plot-container, .stats-panel, .stl-viewer-section {
                padding: 12px;
            }
            #stlContainer {
                height: 280px;
            }
            #plotly-container, #deviation-container {
                height: 360px;
            }
            .main-tab-bar {
                overflow-x: auto;
                -webkit-overflow-scrolling: touch;
            }
            .main-tab-btn {
                flex: 0 0 auto;
                padding: 10px 14px;
                font-size: 12px;
            }
            .tab-bar {
                gap: 8px !important;
            }
            .stat-name {
                min-width: 0;
            }
            .stat-row {
                flex-wrap: wrap;
                gap: 4px;
            }
        }
    </style>
</head>
<body class="light-mode">
    <div class="container">
        <header>
            <h1 id="mainTitle">VMI Simulation Analysis</h1>
            <p id="mainSubtitle">Velocity Map Imaging &mdash; Disturbance Effects Comparison</p>
            <div class="theme-toggle">
                <span class="theme-label">&#127769;</span>
                <label class="toggle-switch">
                    <input type="checkbox" id="themeToggle" onchange="toggleTheme()" checked>
                    <span class="toggle-slider"></span>
                </label>
                <span class="theme-label">&#9728;</span>
            </div>
        </header>        
        
        <div class="header-toolbar" style="display:flex; gap:8px; align-items:center; justify-content:flex-start; margin-top:10px; flex-wrap: wrap;">
            <button class="tab-btn" onclick="downloadDetectorPositions('png')">&#8659; Download Detektorposition</button>
            <button class="tab-btn" onclick="download3D()">&#9632; Export 3D PNG</button>
        </div>

        <div class="main-grid">
            <aside class="sidebar">
                <h3>Model Selection</h3>
                 
                <!-- MODEL UPLOAD SECTION -->
                <div style="background: var(--bg-input); padding: 12px; margin-bottom: 15px; border-radius: 10px; border: 1px solid #667eea;">
                    <h4 style="margin: 0 0 10px; color: #667eea; font-size: 13px; font-weight: 600;">Load Model Folder</h4>
                    <div style="background: var(--bg-card); padding: 10px; border-radius: 6px; margin-bottom: 10px; border: 1px solid var(--border-color);">
                        <p style="margin: 0; color: var(--text-secondary); font-size: 11px;">
                            Select a folder containing: <strong style="color: var(--text-primary);">result</strong> file (required) + optional: trajectory_data, electrode*.stl
                        </p>
                    </div>
                    <div style="display: flex; gap: 5px; margin-bottom: 10px; flex-wrap: wrap;">
                        <input type="text" id="modelNameInput" placeholder="Model name (auto-detected)" style="flex: 1; min-width: 120px; padding: 6px; border: 1px solid var(--border-color); border-radius: 6px; background: var(--bg-card); color: var(--text-primary); font-size: 11px;">
                        <input type="file" id="modelFolderInput" webkitdirectory directory mozdirectory style="flex: 1; min-width: 120px; padding: 6px; border: 1px solid var(--border-color); border-radius: 6px; background: var(--bg-card); color: var(--text-secondary); font-size: 10px;">
                    </div>
                    <button onclick="handleModelUpload()" style="width: 100%; padding: 8px; background: #667eea; border: none; border-radius: 4px; color: white; cursor: pointer; font-weight: 600; font-size: 12px;">Load Folder</button>
                    <div id="uploadStatus" style="display: none; margin-top: 8px; padding: 8px; border-radius: 4px; font-size: 11px;"></div>
                </div>
                 
                <!-- UPLOADED MODELS LIST -->
                <div id="modelSelectionList" style="background: var(--bg-input); padding: 10px; border-radius: 10px; min-height: 40px; border: 1px solid var(--border-color);">
                    <p style="color: var(--text-secondary); margin: 0; font-size: 11px;">No models uploaded yet...</p>
                </div>
                 
                <button class="btn-update" onclick="updateVisibility()" style="margin-top: 10px;">Update Plot</button>
            </aside>

            
            <div class="content-area">
                <div class="stl-viewer-section active" id="stlSection">
                    <div class="stl-viewer-header">
                        <h3 id="stlTitle">3D Model Viewer</h3>
                        <div style="display: flex; align-items: center; gap: 15px;">
                            <label style="display: flex; align-items: center; gap: 6px; color: var(--text-secondary); font-size: 12px;">
                                <input type="checkbox" id="trajShowTrajectories" checked onchange="toggleTrajectories()"> Show Trajectories
                            </label>
                            <button class="btn-close-3d" onclick="closeSTLViewer()">Close</button>
                        </div>
                    </div>
                    <div id="stlContainer" style="height: 420px; display: flex; align-items: center; justify-content: center; color: #666; font-size: 13px;">
                        No model loaded &mdash; upload a folder with STL files to see the 3D view
                    </div>
                    
                    <!-- Clipping Plane Controls -->
                    <div id="clipControls" style="margin-top: 12px; padding: 12px; background: var(--bg-input); border-radius: 10px; border: 1px solid var(--border-color);">
                        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                            <strong style="color: var(--text-primary); font-size: 12px;">Section Planes</strong>
                            <button onclick="resetClipping()" style="margin-left: auto; padding: 4px 10px; background: rgba(100,100,100,0.3); border: 1px solid var(--border-color); border-radius: 4px; color: var(--text-secondary); cursor: pointer; font-size: 10px;">Reset</button>
                        </div>
                        <div style="display: grid; grid-template-columns: 50px 1fr 50px; gap: 6px; align-items: center;">
                            <span style="color: #ef5350; font-size: 11px; font-weight: 500;">X:</span>
                            <input type="range" id="clipX" min="-250" max="250" value="250" style="width: 100%; cursor: pointer;" oninput="updateClipping()">
                            <span id="clipXLabel" style="color: var(--text-secondary); font-size: 10px;">Off</span>
                            
                            <span style="color: #66bb6a; font-size: 11px; font-weight: 500;">Y:</span>
                            <input type="range" id="clipY" min="-100" max="100" value="100" style="width: 100%; cursor: pointer;" oninput="updateClipping()">
                            <span id="clipYLabel" style="color: var(--text-secondary); font-size: 10px;">Off</span>
                            
                            <span style="color: #42a5f5; font-size: 11px; font-weight: 500;">Z:</span>
                            <input type="range" id="clipZ" min="-100" max="100" value="0" style="width: 100%; cursor: pointer;" oninput="updateClipping()">
                            <span id="clipZLabel" style="color: var(--text-secondary); font-size: 10px;">0mm</span>
                        </div>
                    </div>
                    
                    <!-- Trajectory Animation Controls -->
                    <div id="trajControls" style="margin-top: 12px; padding: 12px; background: var(--bg-input); border-radius: 10px; border: 1px solid var(--border-color);">
                        <div style="display: flex; align-items: center; gap: 15px; flex-wrap: wrap;">
                            <button id="trajPlayBtn" onclick="toggleTrajectoryPlay()" style="padding: 8px 20px; background: linear-gradient(135deg, #667eea, #764ba2); border: none; border-radius: 6px; color: white; cursor: pointer; font-weight: 500; font-size: 13px;">&#9654; Play</button>
                            <button onclick="resetTrajectory()" style="padding: 8px 15px; background: rgba(244,67,54,0.8); border: none; border-radius: 6px; color: white; cursor: pointer; font-size: 12px;">Reset</button>
                            <div style="flex: 1; display: flex; align-items: center; gap: 8px; min-width: 200px;">
                                <span style="color: var(--text-secondary); font-size: 11px;">TOF:</span>
                                <input type="range" id="trajScrubber" min="0" max="100" value="0" style="flex: 1; cursor: pointer;" oninput="scrubTrajectory()">
                                <span id="trajTimeLabel" style="color: var(--text-primary); font-size: 11px; min-width: 70px;">0.000 &micro;s</span>
                            </div>
                            <label style="display: flex; align-items: center; gap: 6px; color: var(--text-secondary); font-size: 11px;">
                                Speed:
                                <input type="range" id="trajSpeed" min="0.1" max="5" step="0.1" value="1" style="width: 60px; cursor: pointer;">
                                <span id="trajSpeedLabel" style="min-width: 30px;">1.0x</span>
                            </label>
                        </div>
                        <div style="margin-top: 8px; display: flex; gap: 15px; flex-wrap: wrap;">
                            <label style="display: flex; align-items: center; gap: 5px; color: var(--text-secondary); font-size: 11px;">
                                <input type="checkbox" id="trajShowTrails" checked onchange="updateTrajectoryOptions()"> Trails
                            </label>
                            <label style="display: flex; align-items: center; gap: 5px; color: var(--text-secondary); font-size: 11px;">
                                <input type="checkbox" id="trajColorByKE" onchange="updateTrajectoryOptions()"> Color by KE
                            </label>
                            <span style="color: var(--text-muted); font-size: 10px; margin-left: auto;" id="trajIonCount"></span>
                        </div>
                    </div>
                </div>
                
                <!-- Main Visualization Tabs -->
                <div class="main-tab-bar">
                    <button class="main-tab-btn active" id="main-tab-positions" onclick="switchMainTab('positions')">Positions</button>
                        <button class="main-tab-btn" id="main-tab-tof" onclick="switchMainTab('tof')" style="display:none;">Time of Flight</button>
                        <button class="main-tab-btn" id="main-tab-radial" onclick="switchMainTab('radial')" style="display:none;">Radial Distribution</button>
                        <button class="main-tab-btn" id="main-tab-energy" onclick="switchMainTab('energy')" style="display:none;">Kinetic Energy</button>
                </div>
                
                <!-- Positions Panel -->
                <div class="main-tab-panel active" id="panel-positions">
                    <div class="plot-container">
                        <div class="tab-bar" style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
                            <span style="font-size: 12px; color: var(--text-secondary); font-weight: 500;">Time:</span>
                            <button class="tab-btn" id="btnJumpStart" onclick="jumpToTime(0)" style="padding: 6px 12px;">
                                &#9198; Start
                            </button>
                            <input type="range" id="posTimeScrubber" min="0" max="1000" value="0" 
                                   style="flex: 1; min-width: 150px; cursor: pointer;" 
                                   oninput="scrubPositionTime()">
                            <button class="tab-btn" id="btnJumpEnd" onclick="jumpToTime(-1)" style="padding: 6px 12px;">
                                End &#9197;
                            </button>
                            <span id="posTimeLabel" style="font-size: 12px; color: var(--text-secondary); padding: 6px 12px; background: var(--bg-input); border-radius: 6px; min-width: 80px; text-align: center;">
                                &#9201; 0.000 &micro;s
                            </span>
                        </div>
                        <div id="position-plots-row" class="position-plots-row single">
                            <div class="position-plot-cell">
                                <div id="plotly-container"></div>
                            </div>
                            <div class="position-plot-cell" id="deviation-cell" style="display: none;">
                                <div id="deviation-container"></div>
                            </div>
'''
    
    page_middle = '''
                        </div>
                    </div>
                    <div class="stats-panel" style="margin-top: 15px;">
                        <h3>Deviation Statistics (vs Ideal Reference)</h3>
''' + stats_html + '''
                    </div>
                </div>
                
                <!-- TOF Panel -->
                <div class="main-tab-panel" id="panel-tof">
                    <div class="plot-container">
                        <div id="tof-plot"><!-- TOF_PLOT_PLACEHOLDER --></div>
                    </div>
                </div>
                
                <!-- Radial Panel -->
                <div class="main-tab-panel" id="panel-radial">
                    <div class="plot-container">
                        <div id="radial-plot"><!-- RADIAL_PLOT_PLACEHOLDER --></div>
                    </div>
                </div>
                
                <!-- Energy Panel -->
                <div class="main-tab-panel" id="panel-energy">
                    <div class="plot-container">
                        <div id="energy-plot"><!-- ENERGY_PLOT_PLACEHOLDER --></div>
                    </div>
                </div>
            </div>
        </div>
    </div>
'''
    
    # JavaScript to update trace visibility
    # Build model to trace index mapping
    model_keys_list = list(MODELS.keys())
    model_keys_in_data = [k for k in model_keys_list if k in all_data]
    
    # Load STL files for all electrodes per model and encode as Base64
    stl_data = {}
    model_charges = {}  # Load electrode charges for each model
    for model_key, config in MODELS.items():
        folder = config.get('folder')
        num_electrodes = config.get('num_electrodes', 9)
        if folder and folder.exists():
            electrode_data = []
            total_size = 0
            electrodes_found = 0
            for i in range(1, num_electrodes + 1):
                stl_file = folder / f"electrode{i}.stl"
                if stl_file.exists():
                    with open(stl_file, 'rb') as f:
                        stl_bytes = f.read()
                        electrode_data.append(base64.b64encode(stl_bytes).decode('utf-8'))
                        total_size += len(stl_bytes)
                        electrodes_found += 1
                else:
                    electrode_data.append(None)
            if electrodes_found > 0:
                stl_data[model_key] = electrode_data
                print(f"  [STL] {config['name']}: {electrodes_found}/{num_electrodes} electrodes ({total_size / 1024:.1f} KB)")
        
        # Load electrode charges for this model
        charges = load_electrode_charges(config)
        if charges:
            model_charges[model_key] = charges
    
    # Create JavaScript object with embedded STL data (array of electrodes per model)
    stl_data_js = "var stlDataBase64 = {\n"
    for key, electrode_array in stl_data.items():
        stl_data_js += f"    '{key}': ["
        electrode_strings = []
        for data in electrode_array:
            if data:
                electrode_strings.append(f"'{data}'")
            else:
                electrode_strings.append("null")
        stl_data_js += ", ".join(electrode_strings)
        stl_data_js += "],\n"
    stl_data_js += "};\n"
    
    # Electrode colors for 3D viewer
    electrode_colors_js = "var electrodeColors = ["
    electrode_colors_js += ", ".join([f"0x{c:06x}" for c in ELECTRODE_COLORS])
    electrode_colors_js += "];\n"
    
    # Electrode charges for 3D viewer legend
    import json
    model_charges_js = "var modelCharges = "
    model_charges_js += json.dumps(model_charges) + ";\n"
    # Embed statistics (computed earlier) for client-side downloads &mdash; do not auto-generate files on run
    try:
        stats_js = "var allStats = " + json.dumps(all_stats) + ";\n"
    except Exception:
        stats_js = "var allStats = {};\n"
    
    # Generate trajectory data as JavaScript
    import json
    trajectory_js = "var trajectoryData = "
    trajectory_js_data = {}
    for model_key, traj in trajectory_data.items():
        model_traj = {}
        for ion_n, steps in traj.items():
            model_traj[str(ion_n)] = steps
        trajectory_js_data[model_key] = model_traj
    trajectory_js += json.dumps(trajectory_js_data) + ";\n"
    
    # Model colors for trajectory visualization
    model_colors_js = "var modelColors = {"
    model_colors_js += ", ".join([f"'{k}': '{v['color']}'" for k, v in MODELS.items()])
    model_colors_js += "};\n"
    
    # Plotly and Three.js scripts
    threejs_scripts = '''
    <!-- Plotly for detector position plots -->
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <!-- Three.js and STLLoader from CDN -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/STLLoader.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
'''
    
    js_script = '''
    <script>
    ''' + stl_data_js + electrode_colors_js + model_charges_js + stats_js + trajectory_js + model_colors_js + '''
    var stlScene, stlCamera, stlRenderer, stlControls;
    var clipPlaneX, clipPlaneY, clipPlaneZ;
    var clippingEnabled = false;
    
    function base64ToArrayBuffer(base64) {
        var binaryString = atob(base64);
        var bytes = new Uint8Array(binaryString.length);
        for (var i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }
        return bytes.buffer;
    }
    
    function openSTLViewer(modelKey, modelName) {
        document.getElementById('stlSection').classList.add('active');
        document.getElementById('stlTitle').textContent = '3D Model: ' + modelName;
        
        var container = document.getElementById('stlContainer');
        container.innerHTML = '';
        
        // Check if STL data exists (now an array of electrodes)
        if (!stlDataBase64[modelKey] || !stlDataBase64[modelKey].length) {
            container.innerHTML = '<p style="color: #f44336; text-align: center; padding-top: 100px;">No STL data for this model</p>';
            return;
        }
        
        // Setup Three.js scene with theme-aware background
        stlScene = new THREE.Scene();
        var isLightTheme = document.documentElement.getAttribute('data-theme') === 'light';
        stlScene.background = new THREE.Color(isLightTheme ? 0xf0f0f5 : 0x15151f);
        
        stlCamera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 10000);
        stlCamera.position.set(100, 100, 200);
        
        // Enable preserveDrawingBuffer so canvas.toDataURL works for image export
        stlRenderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true, alpha: false });
        stlRenderer.setSize(container.clientWidth, container.clientHeight);
        // Set clear color to match theme background
        stlRenderer.setClearColor(isLightTheme ? 0xf0f0f5 : 0x15151f, 1);
        stlRenderer.localClippingEnabled = true;
        container.appendChild(stlRenderer.domElement);
        
        // Initialize clipping planes (pointing inward, initially disabled at max values)
        clipPlaneX = new THREE.Plane(new THREE.Vector3(-1, 0, 0), 250);
        clipPlaneY = new THREE.Plane(new THREE.Vector3(0, -1, 0), 100);
        clipPlaneZ = new THREE.Plane(new THREE.Vector3(0, 0, -1), 100);
        
        // Reset clipping sliders
        document.getElementById('clipX').value = 250;
        document.getElementById('clipY').value = 100;
        document.getElementById('clipZ').value = 0;
        document.getElementById('clipXLabel').textContent = 'Off';
        document.getElementById('clipYLabel').textContent = 'Off';
        document.getElementById('clipZLabel').textContent = '0mm';
        
        // Initialize Z clipping plane at 0
        if (clipPlaneZ) clipPlaneZ.constant = 0;
        
        // Add lights
        var ambientLight = new THREE.AmbientLight(0x606060, 0.6);
        stlScene.add(ambientLight);
        var directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(1, 1, 1);
        stlScene.add(directionalLight);
        var directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.5);
        directionalLight2.position.set(-1, -0.5, -1);
        stlScene.add(directionalLight2);
        var directionalLight3 = new THREE.DirectionalLight(0xffffff, 0.3);
        directionalLight3.position.set(0, -1, 0);
        stlScene.add(directionalLight3);
        
        // Add grid
        var gridHelper = new THREE.GridHelper(400, 40);
        stlScene.add(gridHelper);
        
        // Add axes
        var axesHelper = new THREE.AxesHelper(100);
        stlScene.add(axesHelper);
        
        // Orbit controls
        stlControls = new THREE.OrbitControls(stlCamera, stlRenderer.domElement);
        stlControls.enableDamping = true;
        
        // Load all electrodes from Base64 data array
        var loader = new THREE.STLLoader();
        var electrodeData = stlDataBase64[modelKey];
        var globalBoundingBox = new THREE.Box3();
        var meshes = new Array(electrodeData.length).fill(null);  // Index-aligned with electrodeData
        
        try {
            for (var i = 0; i < electrodeData.length; i++) {
                if (electrodeData[i] === null) continue;
                
                var arrayBuffer = base64ToArrayBuffer(electrodeData[i]);
                var geometry = loader.parse(arrayBuffer);
                
                // Use electrode-specific color from palette
                var colorIndex = i % electrodeColors.length;
                var material = new THREE.MeshPhongMaterial({ 
                    color: electrodeColors[colorIndex], 
                    specular: 0x222222, 
                    shininess: 60,
                    side: THREE.DoubleSide,
                    transparent: true,
                    opacity: 0.9,
                    clippingPlanes: [clipPlaneX, clipPlaneY, clipPlaneZ],
                    clipShadows: true
                });
                var mesh = new THREE.Mesh(geometry, material);
                mesh.name = 'electrode' + (i + 1);
                
                // Update global bounding box
                geometry.computeBoundingBox();
                globalBoundingBox.union(geometry.boundingBox);
                
                meshes[i] = mesh;  // Store at correct index
                stlScene.add(mesh);
            }
            
            // Center all meshes based on global bounding box
            var center = new THREE.Vector3();
            globalBoundingBox.getCenter(center);
            meshes.forEach(function(mesh) {
                if (mesh) mesh.position.sub(center);
            });
            
            // Store offset globally for trajectory alignment
            window.electrodeOffset = center.clone();
            
            // Auto-scale camera based on overall size
            var size = new THREE.Vector3();
            globalBoundingBox.getSize(size);
            var maxDim = Math.max(size.x, size.y, size.z);
            stlCamera.position.set(maxDim * 0.8, maxDim * 0.8, maxDim * 1.2);
            stlControls.target.set(0, 0, 0);
            stlControls.update();
            
        } catch (error) {
            container.innerHTML = '<p style="color: #f44336; text-align: center; padding-top: 100px;">Error parsing STL: ' + error + '</p>';
            return;
        }
        
        // Store meshes globally for toggle functionality
        window.stlMeshes = meshes;
        
        // Add interactive electrode legend (theme-aware)
        var isLightTheme = document.documentElement.getAttribute('data-theme') === 'light';
        var legendBg = isLightTheme ? 'rgba(255,255,255,0.9)' : 'rgba(0,0,0,0.85)';
        var legendTextColor = isLightTheme ? '#333' : 'white';
        var legendBorderColor = isLightTheme ? 'rgba(0,0,0,0.2)' : 'rgba(255,255,255,0.3)';
        
        var legendDiv = document.createElement('div');
        legendDiv.style.cssText = 'position: absolute; top: 10px; right: 10px; background: ' + legendBg + '; padding: 12px; border-radius: 8px; font-size: 12px; min-width: 130px;';
        legendDiv.innerHTML = '<strong style="color: ' + legendTextColor + '; display: block; margin-bottom: 8px; border-bottom: 1px solid ' + legendBorderColor + '; padding-bottom: 6px;">Electrodes</strong>';
        
        for (var i = 0; i < electrodeData.length; i++) {
            if (electrodeData[i] !== null) {
                var colorHex = '#' + electrodeColors[i % electrodeColors.length].toString(16).padStart(6, '0');
                var legendItem = document.createElement('div');
                legendItem.className = 'legend-item';
                legendItem.dataset.electrodeIndex = i;
                legendItem.dataset.visible = 'true';
                legendItem.style.cssText = 'cursor: pointer; padding: 4px 6px; margin: 2px 0; border-radius: 4px; transition: all 0.2s ease; display: flex; align-items: center; gap: 6px;';
                legendItem.innerHTML = '<span class="legend-color" style="color: ' + colorHex + '; font-size: 14px;">&#9632;</span><span class="legend-text" style="color: ' + legendTextColor + ';">Electrode ' + (i + 1) + '</span>';
                
                // Add hover effect (theme-aware)
                legendItem.onmouseenter = function() { 
                    var light = document.documentElement.getAttribute('data-theme') === 'light';
                    this.style.background = light ? 'rgba(0,0,0,0.05)' : 'rgba(255,255,255,0.1)'; 
                };
                legendItem.onmouseleave = function() { this.style.background = 'transparent'; };
                
                // Toggle electrode visibility on click
                legendItem.onclick = (function(idx) {
                    return function() {
                        var isVisible = this.dataset.visible === 'true';
                        this.dataset.visible = isVisible ? 'false' : 'true';
                        
                        // Toggle mesh visibility
                        if (window.stlMeshes && window.stlMeshes[idx]) {
                            window.stlMeshes[idx].visible = !isVisible;
                        }
                        
                        // Update legend item style
                        var textSpan = this.querySelector('.legend-text');
                        var colorSpan = this.querySelector('.legend-color');
                        if (isVisible) {
                            textSpan.style.textDecoration = 'line-through';
                            textSpan.style.opacity = '0.4';
                            colorSpan.style.opacity = '0.4';
                        } else {
                            textSpan.style.textDecoration = 'none';
                            textSpan.style.opacity = '1';
                            colorSpan.style.opacity = '1';
                        }
                    };
                })(i);
                
                legendDiv.appendChild(legendItem);
            }
        }
        
        // Add "Show All" / "Hide All" buttons
        var btnContainer = document.createElement('div');
        btnContainer.style.cssText = 'display: flex; gap: 4px; margin-top: 10px; padding-top: 8px; border-top: 1px solid ' + legendBorderColor + ';';
        
        var showAllBtn = document.createElement('button');
        showAllBtn.textContent = 'All';
        showAllBtn.style.cssText = 'flex: 1; padding: 5px; font-size: 10px; background: rgba(56,239,125,0.8); border: none; border-radius: 4px; color: white; cursor: pointer;';
        showAllBtn.onclick = function() {
            var items = legendDiv.querySelectorAll('.legend-item');
            items.forEach(function(item, idx) {
                item.dataset.visible = 'true';
                item.querySelector('.legend-text').style.textDecoration = 'none';
                item.querySelector('.legend-text').style.opacity = '1';
                item.querySelector('.legend-color').style.opacity = '1';
                if (window.stlMeshes && window.stlMeshes[idx]) {
                    window.stlMeshes[idx].visible = true;
                }
            });
        };
        
        var hideAllBtn = document.createElement('button');
        hideAllBtn.textContent = 'None';
        hideAllBtn.style.cssText = 'flex: 1; padding: 5px; font-size: 10px; background: rgba(244,67,54,0.8); border: none; border-radius: 4px; color: white; cursor: pointer;';
        hideAllBtn.onclick = function() {
            var items = legendDiv.querySelectorAll('.legend-item');
            items.forEach(function(item, idx) {
                item.dataset.visible = 'false';
                item.querySelector('.legend-text').style.textDecoration = 'line-through';
                item.querySelector('.legend-text').style.opacity = '0.4';
                item.querySelector('.legend-color').style.opacity = '0.4';
                if (window.stlMeshes && window.stlMeshes[idx]) {
                    window.stlMeshes[idx].visible = false;
                }
            });
        };
        
        btnContainer.appendChild(showAllBtn);
        btnContainer.appendChild(hideAllBtn);
        legendDiv.appendChild(btnContainer);
        
        container.style.position = 'relative';
        container.appendChild(legendDiv);
        
        // Add charges legend if available
        if (modelCharges && modelCharges[modelKey]) {
            var chargesDiv = document.createElement('div');
            chargesDiv.style.cssText = 'position: absolute; top: 10px; left: 10px; background: ' + legendBg + '; padding: 12px; border-radius: 8px; font-size: 12px; max-width: 200px;';
            chargesDiv.innerHTML = '<strong style="color: ' + legendTextColor + '; display: block; margin-bottom: 8px; border-bottom: 1px solid ' + legendBorderColor + '; padding-bottom: 6px;">Electrode Charges</strong>';
            
            var charges = modelCharges[modelKey];
            for (var i = 0; i < charges.length && i < electrodeData.length; i++) {
                if (electrodeData[i] !== null) {
                    var colorHex = '#' + electrodeColors[i % electrodeColors.length].toString(16).padStart(6, '0');
                    var chargeItem = document.createElement('div');
                    chargeItem.style.cssText = 'padding: 4px 6px; margin: 2px 0; border-radius: 4px; display: flex; align-items: center; gap: 6px;';
                    chargeItem.innerHTML = '<span style="color: ' + colorHex + '; font-size: 14px;">&#9632;</span><span style="color: ' + legendTextColor + ';">E' + (i + 1) + ': ' + charges[i] + '</span>';
                    chargesDiv.appendChild(chargeItem);
                }
            }
            container.appendChild(chargesDiv);
        }
        
        // Store current model key for trajectory loading
        window.currentSTLModelKey = modelKey;
        
        // Animation loop
        function animate() {
            if (document.getElementById('stlSection').classList.contains('active')) {
                requestAnimationFrame(animate);
                stlControls.update();
                stlRenderer.render(stlScene, stlCamera);
            }
        }
        animate();
        
        // Load trajectories for this model
        loadTrajectoriesForModel(modelKey);
    }
    
    function closeSTLViewer() {
        // Stop trajectory animation
        trajAnimating = false;
        
        document.getElementById('stlSection').classList.remove('active');
        var container = document.getElementById('stlContainer');
        container.innerHTML = '';
        if (stlRenderer) {
            stlRenderer.dispose();
        }
        
        // Clear trajectory data
        trajIonMeshes = [];
        trajTrailLines = [];
    }
    
    // Position plot time controls
    var posMaxTime = 0;
    var posCurrentTime = 0;
    
    // Initialize position time scrubber with max time from trajectory data
    function initPositionTimeScrubber() {
        var modelKeys = ''' + str(model_keys_in_data) + ''';
        posMaxTime = 0;
        
        // Find max time across all models
        modelKeys.forEach(function(key) {
            if (trajectoryData[key]) {
                Object.keys(trajectoryData[key]).forEach(function(ionKey) {
                    var steps = trajectoryData[key][ionKey];
                    if (steps && steps.length > 0) {
                        var lastTof = steps[steps.length - 1].tof;
                        if (lastTof > posMaxTime) posMaxTime = lastTof;
                    }
                });
            }
        });
        
        var scrubber = document.getElementById('posTimeScrubber');
        if (scrubber && posMaxTime > 0) {
            scrubber.max = posMaxTime * 1000;  // Store in ms for precision
            scrubber.value = posMaxTime * 1000;  // Start at end position (like original default)
        }
        
        // Show end positions initially
        jumpToTime(-1);
    }
    
    function scrubPositionTime() {
        var scrubber = document.getElementById('posTimeScrubber');
        if (scrubber) {
            posCurrentTime = parseFloat(scrubber.value) / 1000;
            updatePositionTimeLabel();
            update2DTimePositions(posCurrentTime);
            
            // Also sync 3D viewer if open
            if (trajCurrentModel) {
                trajCurrentTime = posCurrentTime;
                updateTrajectoryPositions(trajCurrentTime);
                var trajScrubber = document.getElementById('trajScrubber');
                var trajLabel = document.getElementById('trajTimeLabel');
                if (trajScrubber) trajScrubber.value = trajCurrentTime * 1000;
                if (trajLabel) trajLabel.textContent = trajCurrentTime.toFixed(3) + ' &micro;s';
            }
        }
    }
    
    function jumpToTime(time) {
        if (time === -1) {
            // Jump to end
            posCurrentTime = posMaxTime;
        } else {
            posCurrentTime = time;
        }
        
        var scrubber = document.getElementById('posTimeScrubber');
        if (scrubber) {
            scrubber.value = posCurrentTime * 1000;
        }
        
        updatePositionTimeLabel();
        update2DTimePositions(posCurrentTime);
        
        // Also sync 3D viewer if open
        if (trajCurrentModel) {
            trajCurrentTime = posCurrentTime;
            updateTrajectoryPositions(trajCurrentTime);
            var trajScrubber = document.getElementById('trajScrubber');
            var trajLabel = document.getElementById('trajTimeLabel');
            if (trajScrubber) trajScrubber.value = trajCurrentTime * 1000;
            if (trajLabel) trajLabel.textContent = trajCurrentTime.toFixed(3) + ' &micro;s';
        }
    }
    
    function updatePositionTimeLabel() {
        var label = document.getElementById('posTimeLabel');
        if (label) {
            if (posCurrentTime <= 0.001) {
                label.innerHTML = '&#9201; <b>START</b>';
            } else if (posCurrentTime >= posMaxTime - 0.001) {
                label.innerHTML = '&#9201; <b>END</b>';
            } else {
                label.innerHTML = '&#9201; ' + posCurrentTime.toFixed(3) + ' &micro;s';
            }
        }
    }
    
    // Initialize on page load
    document.addEventListener('DOMContentLoaded', initPositionTimeScrubber);
    
    // Update 2D position plot with positions at a specific time
    function update2DTimePositions(time) {
        var plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
        if (!plotDiv) return;
        
        var modelKeys = ''' + str(model_keys_in_data) + ''';
        var refKey = 'normal';
        var numTestModels = modelKeys.length - 1;
        var tracesPerType = 4 + 1 + 2 * numTestModels;
        
        // Get positions from trajectory at current time
        function getPositionAtTime(ionSteps, targetTime) {
            if (!ionSteps || ionSteps.length === 0) return null;
            
            // Find timestep closest to target time
            for (var i = 0; i < ionSteps.length; i++) {
                if (ionSteps[i].tof >= targetTime) {
                    return ionSteps[i];
                }
            }
            // Return last position if past end
            return ionSteps[ionSteps.length - 1];
        }
        
        // Get which models are checked
        var checkedModels = {};
        modelKeys.forEach(function(key) {
            var cb = document.getElementById('cb_' + key);
            checkedModels[key] = cb ? cb.checked : false;
        });
        
        // Get reference positions at time
        var refTrajData = trajectoryData['normal'];
        var refYPositions = [];
        var refZPositions = [];
        
        if (refTrajData) {
            Object.keys(refTrajData).sort(function(a,b){return parseInt(a)-parseInt(b);}).forEach(function(ionKey) {
                var pos = getPositionAtTime(refTrajData[ionKey], time);
                if (pos) {
                    refYPositions.push(pos.y);
                    refZPositions.push(pos.z);
                }
            });
        }
        
        // Build visibility - show center lines, reference if checked, and all checked models
        var visibility = [];
        for (var i = 0; i < tracesPerType * 2; i++) {
            visibility.push(false);  // Hide all initially
        }
        
        // Show center lines (first 4 traces)
        visibility[0] = visibility[1] = visibility[2] = visibility[3] = true;
        
        // Show reference trace (trace index 4) if checked
        visibility[4] = checkedModels[refKey] !== false;
        
        // Update reference positions
        if (refYPositions.length > 0) {
            Plotly.restyle(plotDiv, {
                x: [refYPositions],
                y: [refZPositions]
            }, [4]);
        }
        
        // Update each checked test model
        var testModelIdx = 0;
        modelKeys.forEach(function(modelKey) {
            if (modelKey === refKey) return;
            
            var overlayIdx = 5 + testModelIdx * 2;
            var diffIdx = 5 + testModelIdx * 2 + 1;
            
            if (checkedModels[modelKey] && trajectoryData[modelKey]) {
                visibility[overlayIdx] = true;
                visibility[diffIdx] = true;
                
                // Get model positions at time
                var trajData = trajectoryData[modelKey];
                var yPositions = [];
                var zPositions = [];
                
                Object.keys(trajData).sort(function(a,b){return parseInt(a)-parseInt(b);}).forEach(function(ionKey) {
                    var pos = getPositionAtTime(trajData[ionKey], time);
                    if (pos) {
                        yPositions.push(pos.y);
                        zPositions.push(pos.z);
                    }
                });
                
                // Calculate differences
                var diffY = [];
                var diffZ = [];
                var minLen = Math.min(yPositions.length, refYPositions.length);
                for (var i = 0; i < minLen; i++) {
                    diffY.push(yPositions[i] - refYPositions[i]);
                    diffZ.push(zPositions[i] - refZPositions[i]);
                }
                
                // Update overlay trace
                Plotly.restyle(plotDiv, {
                    x: [yPositions],
                    y: [zPositions]
                }, [overlayIdx]);
                
                // Update difference trace
                Plotly.restyle(plotDiv, {
                    x: [diffY],
                    y: [diffZ]
                }, [diffIdx]);
            }
            
            testModelIdx++;
        });
        
        Plotly.restyle(plotDiv, {'visible': visibility});
    }
    
    function updateVisibility() {
        // Update 2D position plot with current time when model selection changes
        update2DTimePositions(posCurrentTime);
    }
    
    // Theme toggle function
    function toggleTheme() {
        var html = document.documentElement;
        var isLight = document.getElementById('themeToggle').checked;
        html.setAttribute('data-theme', isLight ? 'light' : 'dark');
        
        // Toggle light-mode class on body for brainrot elements
        if (isLight) {
            document.body.classList.add('light-mode');
        } else {
            document.body.classList.remove('light-mode');
        }
        
        // Update all Plotly charts colors
        applyPlotlyTheme();
        
        // Update STL viewer background if active
        if (stlScene) {
            stlScene.background = new THREE.Color(isLight ? 0xf0f0f5 : 0x15151f);
        }
        
        // Update STL legend colors
        var stlContainer = document.getElementById('stlContainer');
        if (stlContainer) {
            var legendDiv = stlContainer.querySelector('div[style*="position: absolute"]');
            if (legendDiv) {
                legendDiv.style.background = isLight ? 'rgba(255,255,255,0.9)' : 'rgba(0,0,0,0.85)';
                // Update legend title
                var strongEl = legendDiv.querySelector('strong');
                if (strongEl) {
                    strongEl.style.color = isLight ? '#333' : 'white';
                    strongEl.style.borderBottomColor = isLight ? 'rgba(0,0,0,0.2)' : 'rgba(255,255,255,0.3)';
                }
                // Update legend items text color
                var legendTexts = legendDiv.querySelectorAll('.legend-text');
                legendTexts.forEach(function(textEl) {
                    textEl.style.color = isLight ? '#333' : 'white';
                });
                // Update button container border
                var btnContainer = legendDiv.querySelector('div[style*="border-top"]');
                if (btnContainer) {
                    btnContainer.style.borderTopColor = isLight ? 'rgba(0,0,0,0.2)' : 'rgba(255,255,255,0.3)';
                }
            }
        }
    }
    
    // ==================== TRAJECTORY ANIMATION (Integrated in STL Viewer) ====================
    var trajIonMeshes = [];
    var trajTrailLines = [];
    var trajAnimating = false;
    var trajCurrentTime = 0;
    var trajMaxTime = 0;
    var trajSpeed = 1.0;
    var trajCurrentModel = null;
    var trajVisible = true;
    
    function loadTrajectoriesForModel(modelKey) {
        trajCurrentModel = modelKey;
        
        // Clear existing trajectories
        trajIonMeshes.forEach(function(mesh) { if (stlScene) stlScene.remove(mesh); });
        trajTrailLines.forEach(function(line) { if (stlScene) stlScene.remove(line); });
        trajIonMeshes = [];
        trajTrailLines = [];
        
        if (!trajectoryData[modelKey]) {
            document.getElementById('trajIonCount').textContent = 'No trajectory data';
            return;
        }
        
        var data = trajectoryData[modelKey];
        var ionColor = new THREE.Color(modelColors[modelKey] || '#00d4ff');
        
        // Find max time
        trajMaxTime = 0;
        var ionCount = 0;
        Object.keys(data).forEach(function(ionKey) {
            var steps = data[ionKey];
            ionCount++;
            if (steps.length > 0) {
                var lastTof = steps[steps.length - 1].tof;
                if (lastTof > trajMaxTime) trajMaxTime = lastTof;
            }
        });
        
        // Create ion spheres and trail lines
        Object.keys(data).forEach(function(ionKey) {
            var steps = data[ionKey];
            if (steps.length === 0) return;
             
            // Ion sphere - smaller size to fit with electrodes
            var geometry = new THREE.SphereGeometry(0.8, 12, 12);
            var material = new THREE.MeshPhongMaterial({ 
                color: ionColor, 
                emissive: ionColor, 
                emissiveIntensity: 0.5 
            });
            var sphere = new THREE.Mesh(geometry, material);
            sphere.userData.ionKey = ionKey;
            sphere.userData.steps = steps;
            sphere.visible = false;
            stlScene.add(sphere);
            trajIonMeshes.push(sphere);
             
            // Trail line with per-vertex color support for KE heatmap
            var trailGeometry = new THREE.BufferGeometry();
            var positions = new Float32Array(steps.length * 3);
            var colors = new Float32Array(steps.length * 3);
             
            // Initialize colors based on KE
            var ionColorRGB = ionColor;
            for (var j = 0; j < steps.length; j++) {
                var ke = steps[j].ke || 0.5;  // fallback if no KE data
                var keLog = Math.log10(Math.max(ke, 0.01));
                // Adjusted scale: -1 to 2 (more sensitive for lower KE range)
                var keNorm = Math.min(Math.max((keLog + 1) / 3, 0), 1);
                  
                var r, g, b;
                if (keNorm < 0.25) {
                    var t = keNorm / 0.25;
                    r = 0.0; g = t * 0.8; b = 0.8 + t * 0.2;
                } else if (keNorm < 0.5) {
                    var t = (keNorm - 0.25) / 0.25;
                    r = 0.0; g = 0.8 + t * 0.2; b = 1.0 - t * 1.0;
                } else if (keNorm < 0.75) {
                    var t = (keNorm - 0.5) / 0.25;
                    r = t * 1.0; g = 1.0; b = 0.0;
                } else {
                    var t = (keNorm - 0.75) / 0.25;
                    r = 1.0; g = 1.0 - t * 1.0; b = 0.0;
                }
                colors[j * 3] = r;
                colors[j * 3 + 1] = g;
                colors[j * 3 + 2] = b;
            }
             
            trailGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            trailGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
            trailGeometry.setDrawRange(0, 0);
             
            var trailMaterial = new THREE.LineBasicMaterial({ 
                vertexColors: true,
                opacity: 0.8, 
                transparent: true 
            });
            var trailLine = new THREE.Line(trailGeometry, trailMaterial);
            trailLine.userData.ionKey = ionKey;
            trailLine.userData.steps = steps;
            stlScene.add(trailLine);
            trajTrailLines.push(trailLine);
        });
         
        // Update UI
        document.getElementById('trajIonCount').textContent = ionCount + ' ions, ' + trajMaxTime.toFixed(2) + ' &micro;s';
        document.getElementById('trajScrubber').max = trajMaxTime * 1000;
        document.getElementById('trajScrubber').value = 0;
        document.getElementById('trajTimeLabel').textContent = '0.000 &micro;s';
         
        // Reset and show initial state
        trajCurrentTime = 0;
        trajAnimating = false;
        document.getElementById('trajPlayBtn').textContent = '&#9654; Play';
        updateTrajectoryPositions(0);
    }
    
    function updateClipping() {
        var xVal = parseFloat(document.getElementById('clipX').value);
        var yVal = parseFloat(document.getElementById('clipY').value);
        var zVal = parseFloat(document.getElementById('clipZ').value);
        
        // Update clipping plane positions
        if (clipPlaneX) clipPlaneX.constant = xVal;
        if (clipPlaneY) clipPlaneY.constant = yVal;
        if (clipPlaneZ) clipPlaneZ.constant = zVal;
        
        // Update labels
        var maxX = 250, maxY = 100, maxZ = 100;
        document.getElementById('clipXLabel').textContent = xVal >= maxX ? 'Off' : xVal.toFixed(0) + 'mm';
        document.getElementById('clipYLabel').textContent = yVal >= maxY ? 'Off' : yVal.toFixed(0) + 'mm';
        document.getElementById('clipZLabel').textContent = zVal >= maxZ ? 'Off' : zVal.toFixed(0) + 'mm';
    }
    
    function resetClipping() {
        document.getElementById('clipX').value = 250;
        document.getElementById('clipY').value = 100;
        document.getElementById('clipZ').value = 100;
        updateClipping();
    }
    
    function toggleTrajectories() {
        var checkbox = document.getElementById('trajShowTrajectories');
        trajVisible = checkbox ? checkbox.checked : true;
        
        trajIonMeshes.forEach(function(mesh) {
            if (trajVisible) {
                // Visibility controlled by updateTrajectoryPositions
                updateTrajectoryPositions(trajCurrentTime);
            } else {
                mesh.visible = false;
            }
        });
        trajTrailLines.forEach(function(line) {
            line.visible = trajVisible && document.getElementById('trajShowTrails').checked;
        });
    }
    
    function updateTrajectoryPositions(time) {
        try {
            if (!trajVisible) return;
            
            var showTrails = document.getElementById('trajShowTrails') ? document.getElementById('trajShowTrails').checked : true;
            var colorByKE = document.getElementById('trajColorByKE') ? document.getElementById('trajColorByKE').checked : false;
            
            // Safety: ensure arrays align
            if (trajTrailLines.length !== trajIonMeshes.length) {
                    console.warn('Mismatched traj arrays:', trajIonMeshes.length, trajTrailLines.length);
            }
            
            trajIonMeshes.forEach(function(mesh, idx) {
                    var steps = mesh.userData.steps;
                    var trail = trajTrailLines[idx];
                    if (!steps) return;
                
                    // Find the appropriate timestep for this time
                    var stepIdx = 0;
                    for (var i = 0; i < steps.length; i++) {
                        if (steps[i].tof <= time) {
                            stepIdx = i;
                        } else {
                            break;
                        }
                    }
                
                    // Check if ion has been created yet
                    if (time < steps[0].tof) {
                        mesh.visible = false;
                        if (trail) try { trail.geometry.setDrawRange(0, 0); } catch(e){}
                        return;
                    }
                
                    var step = steps[stepIdx];
                    var offset = window.electrodeOffset || new THREE.Vector3();
                    // SIMION origin is at electrode corner - need additional 85mm offset for Y and Z
                    var simionOffsetY = 85;
                    var simionOffsetZ = 85;
                    mesh.position.set(step.x - offset.x, step.y - offset.y - simionOffsetY, step.z - offset.z - simionOffsetZ);
                    mesh.visible = trajVisible;
                
                    // Update color by KE if enabled - using hot colormap (blue -> cyan -> green -> yellow -> red)
                    if (colorByKE) {
                        try {
                            var keLog = Math.log10(Math.max(step.ke, 0.01));  // log10 of KE
                            // Adjusted scale: -1 to 2 (more sensitive for lower KE range)
                            var keNorm = Math.min(Math.max((keLog + 1) / 3, 0), 1);
                        
                            var r, g, b;
                            if (keNorm < 0.25) {
                                var t = keNorm / 0.25;
                                r = 0.0; g = t * 0.8; b = 0.8 + t * 0.2;
                            } else if (keNorm < 0.5) {
                                var t = (keNorm - 0.25) / 0.25;
                                r = 0.0; g = 0.8 + t * 0.2; b = 1.0 - t * 1.0;
                            } else if (keNorm < 0.75) {
                                var t = (keNorm - 0.5) / 0.25;
                                r = t * 1.0; g = 1.0; b = 0.0;
                            } else {
                                var t = (keNorm - 0.75) / 0.25;
                                r = 1.0; g = 1.0 - t * 1.0; b = 0.0;
                            }
                            mesh.material.color.setRGB(r, g, b);
                            mesh.material.emissive.setRGB(r * 0.3, g * 0.3, b * 0.3);
                        } catch(e) {
                            // ignore per-mesh color errors
                        }
                    }
                
            // Update trail
            if (trail) {
                if (showTrails && trajVisible) {
                    var positions = trail.geometry.attributes.position.array;
                    var colors = trail.geometry.attributes.color ? trail.geometry.attributes.color.array : null;
                    var offset = window.electrodeOffset || new THREE.Vector3();
                    var simionOffsetY = 85;
                    var simionOffsetZ = 85;
                    
                    for (var i = 0; i <= stepIdx; i++) {
                        // Update position
                        positions[i * 3] = steps[i].x - offset.x;
                        positions[i * 3 + 1] = steps[i].y - offset.y - simionOffsetY;
                        positions[i * 3 + 2] = steps[i].z - offset.z - simionOffsetZ;
                        
                        // Update color if available and KE coloring disabled
                        // (KE colors are pre-computed during geometry creation)
                        if (colors && !colorByKE) {
                            // Use model color for all vertices
                            var ionColorRGB = new THREE.Color(modelColors[trajCurrentModel] || '#00d4ff');
                            colors[i * 3] = ionColorRGB.r;
                            colors[i * 3 + 1] = ionColorRGB.g;
                            colors[i * 3 + 2] = ionColorRGB.b;
                        }
                    }
                    
                    trail.geometry.attributes.position.needsUpdate = true;
                    if (colors) trail.geometry.attributes.color.needsUpdate = true;
                    trail.geometry.setDrawRange(0, stepIdx + 1);
                    trail.visible = true;
                } else {
                    trail.geometry.setDrawRange(0, 0);
                    trail.visible = false;
                }
            }
            });
        } catch (e) {
            console.error('updateTrajectoryPositions error', e);
        }
    }
    
    function toggleTrajectoryPlay() {
        trajAnimating = !trajAnimating;
        var btn = document.getElementById('trajPlayBtn');
        if (btn) btn.textContent = trajAnimating ? '&#9208; Pause' : '&#9654; Play';
        
        if (trajAnimating) {
            var speedEl = document.getElementById('trajSpeed');
            trajSpeed = speedEl ? parseFloat(speedEl.value) : 1.0;
            lastTrajTime = performance.now();
            animateTrajectory();
        }
    }
    
    var lastTrajTime = 0;
    function animateTrajectory() {
        if (!trajAnimating) return;
        
        var now = performance.now();
        var delta = (now - lastTrajTime) / 1000;
        lastTrajTime = now;
        
        trajCurrentTime += delta * trajSpeed;
        if (trajCurrentTime > trajMaxTime) {
            trajCurrentTime = trajMaxTime;
            trajAnimating = false;
            var btn = document.getElementById('trajPlayBtn');
            if (btn) btn.textContent = '&#9654; Play';
        }
        
        updateTrajectoryPositions(trajCurrentTime);
        
        // Sync 2D position plot
        update2DTimePositions(trajCurrentTime);
        
        // Update 3D UI
        var scrubber = document.getElementById('trajScrubber');
        var label = document.getElementById('trajTimeLabel');
        if (scrubber) scrubber.value = trajCurrentTime * 1000;
        if (label) label.textContent = trajCurrentTime.toFixed(3) + ' &micro;s';
        
        // Sync 2D position scrubber
        var posScrubber = document.getElementById('posTimeScrubber');
        if (posScrubber) posScrubber.value = trajCurrentTime * 1000;
        posCurrentTime = trajCurrentTime;
        updatePositionTimeLabel();
        
        if (trajAnimating) {
            requestAnimationFrame(animateTrajectory);
        }
    }
    
    function scrubTrajectory() {
        var scrubber = document.getElementById('trajScrubber');
        if (scrubber) {
            trajCurrentTime = parseFloat(scrubber.value) / 1000;
            updateTrajectoryPositions(trajCurrentTime);
            update2DTimePositions(trajCurrentTime);  // Sync 2D plot
            var label = document.getElementById('trajTimeLabel');
            if (label) label.textContent = trajCurrentTime.toFixed(3) + ' &micro;s';
            
            // Sync 2D position scrubber
            var posScrubber = document.getElementById('posTimeScrubber');
            if (posScrubber) posScrubber.value = trajCurrentTime * 1000;
            posCurrentTime = trajCurrentTime;
            updatePositionTimeLabel();
        }
    }
    
    function resetTrajectory() {
        trajAnimating = false;
        trajCurrentTime = 0;
        var btn = document.getElementById('trajPlayBtn');
        var scrubber = document.getElementById('trajScrubber');
        var label = document.getElementById('trajTimeLabel');
        if (btn) btn.textContent = '&#9654; Play';
        if (scrubber) scrubber.value = 0;
        if (label) label.textContent = '0.000 &micro;s';
        updateTrajectoryPositions(0);
        update2DTimePositions(0);  // Sync 2D plot
        
        // Sync 2D position scrubber
        var posScrubber = document.getElementById('posTimeScrubber');
        if (posScrubber) posScrubber.value = 0;
        posCurrentTime = 0;
        updatePositionTimeLabel();
    }
    
    function updateTrajectoryOptions() {
        updateTrajectoryPositions(trajCurrentTime);
    }
    
    // Speed slider handler
    document.addEventListener('DOMContentLoaded', function() {
        var speedSlider = document.getElementById('trajSpeed');
        if (speedSlider) {
            speedSlider.addEventListener('input', function() {
                trajSpeed = parseFloat(this.value);
                var label = document.getElementById('trajSpeedLabel');
                if (label) label.textContent = trajSpeed.toFixed(1) + 'x';
            });
        }
    });
    
    // Main tab switching (Positions, TOF, Radial, Energy)
    var currentMainTab = 'positions';
    
    function switchMainTab(tabName) {
        currentMainTab = tabName;
        
        // Update tab button styles and panel visibility
        ['positions', 'tof', 'radial', 'energy'].forEach(function(t) {
            var btn = document.getElementById('main-tab-' + t);
            if (btn) btn.classList.toggle('active', t === tabName);
            var panel = document.getElementById('panel-' + t);
            if (panel) {
                panel.classList.toggle('active', t === tabName);
                panel.classList.toggle('hidden', t !== tabName);
            }
        });
        
        // Resize Plotly charts in the newly visible panel (fixes rendering issues)
        setTimeout(function() {
            var activePanel = document.getElementById('panel-' + tabName);
            if (activePanel) {
                var plots = activePanel.querySelectorAll('.plotly-graph-div');
                plots.forEach(function(plot) {
                    Plotly.Plots.resize(plot);
                });
            }
        }, 50);
    }
    
    // Initialize all plots on load (render visible first, then hide)
    document.addEventListener('DOMContentLoaded', function() {
        // All panels start visible for initial Plotly render, then hide non-active
        var panels = ['tof', 'radial', 'energy'];
        
        // Brief delay to ensure Plotly has rendered
        setTimeout(function() {
            panels.forEach(function(p) {
                var panel = document.getElementById('panel-' + p);
                if (panel && !panel.classList.contains('active')) {
                    var plots = panel.querySelectorAll('.plotly-graph-div');
                    plots.forEach(function(plot) {
                        Plotly.Plots.resize(plot);
                    });
                    // Now hide
                    panel.classList.add('hidden');
                }
            });
            
            // Apply theme colors to Plotly charts on initial load
            applyPlotlyTheme();
        }, 200);
    });
    
    // Function to apply current theme to all Plotly charts
    function applyPlotlyTheme() {
        var isLight = document.getElementById('themeToggle').checked;
        var plots = document.getElementsByClassName('plotly-graph-div');
        var bgColor = isLight ? 'rgba(250,250,255,0.9)' : 'rgba(25,25,35,0.8)';
        var gridColor = isLight ? 'rgba(0,0,0,0.1)' : 'rgba(100,100,120,0.3)';
        var fontColor = isLight ? '#333' : '#aaa';
        
        for (var i = 0; i < plots.length; i++) {
            Plotly.relayout(plots[i], {
                'paper_bgcolor': 'rgba(0,0,0,0)',
                'plot_bgcolor': bgColor,
                'xaxis.gridcolor': gridColor,
                'yaxis.gridcolor': gridColor,
                'xaxis2.gridcolor': gridColor,
                'yaxis2.gridcolor': gridColor,
                'xaxis.tickfont.color': fontColor,
                'yaxis.tickfont.color': fontColor,
                'xaxis2.tickfont.color': fontColor,
                'yaxis2.tickfont.color': fontColor,
                'legend.font.color': fontColor,
                'legend.bgcolor': isLight ? 'rgba(255,255,255,0.9)' : 'rgba(30,30,40,0.9)'
            });
        }
    }

        // Publication export helpers
        function downloadDataUrl(dataUrl, filename) {
            var a = document.createElement('a');
            a.href = dataUrl;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            a.remove();
        }

        function downloadDetectorPositions(format) {
            // Get selected models
            var modelKeys = window.modelKeys || [];
            var selectedModels = [];
            for (var i = 0; i < modelKeys.length; i++) {
                var k = modelKeys[i];
                var cb = document.getElementById('cb_' + k);
                if (cb && cb.checked) { 
                    selectedModels.push(k); 
                }
            }
            if (selectedModels.length === 0) selectedModels = Object.keys(window.uploadedModelData || {});

            // Request matplotlib export from server
            if (window.location.protocol && window.location.protocol.indexOf('http') === 0) {
                fetch('/export_matplotlib_detector', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({models: selectedModels, format: format, dpi: 300})
                }).then(function(resp){ 
                    if (!resp.ok) throw new Error('Export failed: ' + resp.status);
                    return resp.blob();
                }).then(function(blob){
                    var url = URL.createObjectURL(blob);
                    var filename = 'vmi_detector_positions.' + format;
                    var a = document.createElement('a');
                    a.href = url;
                    a.download = filename;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    setTimeout(function(){ URL.revokeObjectURL(url); }, 10000);
                }).catch(function(err){
                    alert('Download fehlgeschlagen: ' + err);
                    console.error(err);
                });
            } else {
                alert('Download wird nur im HTTP-Modus unterstützt');
            }
        }
        
        function downloadPositions(format) {
            // Legacy function for backward compatibility
            downloadDetectorPositions(format);
        }

        function download3D() {
            var canvas = document.querySelector('#stlContainer canvas');
            if (!canvas) { alert('3D view not open'); return; }
            // Compose a temporary canvas so we can add a legend overlay (charges)
            var w = canvas.width, h = canvas.height;
            var tmp = document.createElement('canvas'); tmp.width = w; tmp.height = h;
            var ctx = tmp.getContext('2d');
            // copy original
            ctx.drawImage(canvas, 0, 0);
            // draw charges legend if available
            var modelKey = (typeof currentModelKey !== 'undefined') ? currentModelKey : (window.trajCurrentModel || null);
            var charges = (typeof modelCharges !== 'undefined' && modelKey && modelCharges[modelKey]) ? modelCharges[modelKey] : null;
            if (charges) {
                var pad = 12, lineH = 18;
                var boxW = 220, boxH = pad*2 + lineH * (charges.length + 1);
                ctx.globalAlpha = 0.85; ctx.fillStyle = 'rgba(0,0,0,0.6)'; ctx.fillRect(10, 10, boxW, boxH); ctx.globalAlpha = 1;
                ctx.fillStyle = 'white'; ctx.font = '14px sans-serif';
                ctx.fillText('Electrode charges', 16, 10 + pad + 12);
                for (var i = 0; i < charges.length; i++) {
                    var y = 10 + pad + 12 + (i+1)*lineH;
                    var colorHex = '#cccccc';
                    try { colorHex = '#' + (electrodeColors[i % electrodeColors.length].toString(16)).padStart(6, '0'); } catch (e) {}
                    ctx.fillStyle = colorHex; ctx.fillRect(16, y-10, 10, 10);
                    ctx.fillStyle = 'white'; ctx.fillText('E' + (i+1) + ': ' + charges[i], 34, y);
                }
            }
            // add annotation
            ctx.fillStyle = 'white'; ctx.font = '16px sans-serif';
            ctx.fillText('0 V region &rarr; increasingly negative potentials', 10, h - 20);
            var dataUrl = tmp.toDataURL('image/png');
            downloadDataUrl(dataUrl, 'vmi_3d.png');
        }


        // ===== MODEL UPLOAD HANDLERS =====
        async function handleModelUpload() {
            const modelName = document.getElementById('modelNameInput').value.trim();
            const files = document.getElementById('modelFolderInput').files;
            const statusDiv = document.getElementById('uploadStatus');
            
            if (files.length === 0) {
                statusDiv.style.display = 'block';
                statusDiv.innerHTML = '<span style="color: #ff6b6b;">&#10060; No folder selected</span>';
                return;
            }
            
            const formData = new FormData();
            if (modelName) formData.append('model_name', modelName);
            
            // Add all files from folder
            for (let file of files) {
                formData.append('files', file);
            }
            
            statusDiv.style.display = 'block';
            statusDiv.innerHTML = '<span style="color: #ffd93d;">&#10231; Loading...</span>';
            
            try {
                console.log('Sending upload request to /upload_models');
                const response = await fetch('/upload_models', {
                    method: 'POST',
                    body: formData
                });
                
                console.log('Response status:', response.status);
                
                let data;
                try {
                    data = await response.json();
                } catch (parseErr) {
                    console.error('Failed to parse response:', parseErr);
                    statusDiv.innerHTML = '<span style="color: #ff6b6b;">&#10060; Server error (invalid response)</span>';
                    return;
                }
                
                if (response.ok) {
                    statusDiv.innerHTML = '<span style="color: #51cf66;">&#10003; Loaded successfully!</span>';
                    document.getElementById('modelNameInput').value = '';
                    document.getElementById('modelFolderInput').value = '';
                    
                    // Refresh uploaded models list
                    setTimeout(loadUploadedModels, 500);
                } else {
                    statusDiv.innerHTML = '<span style="color: #ff6b6b;">&#10060; ' + (data.error || 'Load failed') + '</span>';
                    if (data.warnings && data.warnings.length > 0) {
                        statusDiv.innerHTML += '<div style="font-size: 12px; color: #ffd93d; margin-top: 5px;">' + data.warnings.join('<br>') + '</div>';
                    }
                }
            } catch (err) {
                console.error('Upload error:', err);
                statusDiv.innerHTML = '<span style="color: #ff6b6b;">&#10060; Error: ' + err.message + '</span>';
            }
        }
        
        // Auto-fill model name from folder name
        document.getElementById('modelFolderInput').addEventListener('change', function(e) {
            if (this.files.length > 0) {
                // Get the folder path from first file
                const firstFile = this.files[0];
                const filePath = firstFile.webkitRelativePath || firstFile.name;
                const folderName = filePath.split('/')[0];
                
                if (folderName && !document.getElementById('modelNameInput').value.trim()) {
                    document.getElementById('modelNameInput').value = folderName;
                }
            }
        });
        
        // ===== UPLOADED MODEL PLOT STATE =====
        window.uploadedModelData = {};   // key -> {y:[], z:[], color, name}
        window.stlDataCache = {};        // key -> [b64, b64, ...]  (per electrode)

        async function openSTLViewerUploaded(modelKey, modelName) {
            // Fetch STL data from server if not cached
            if (!window.stlDataCache[modelKey]) {
                try {
                    const resp = await fetch('/get_stl_data/' + modelKey);
                    if (resp.ok) {
                        const d = await resp.json();
                        window.stlDataCache[modelKey] = d.stl_array || [];
                    }
                } catch(e) {
                    alert('Fehler beim Laden der 3D-Daten: ' + e);
                    return;
                }
            }
            // Set as reference for deviation plot
            window.activeRefKey = modelKey;
            // Temporarily inject into stlDataBase64 so openSTLViewer can find it
            stlDataBase64[modelKey] = window.stlDataCache[modelKey];
            openSTLViewer(modelKey, modelName);
            // Re-render deviation plot with new reference
            updateVisibility();
        }

        // Fixed axis range, computed once from end-position data after upload
        window.plotAxisRange = null;
        // Key of the model currently shown in 3D viewer (used as reference for deviation)
        window.activeRefKey = null;

        function getPositionAtTime(ionSteps, targetTime) {
            if (!ionSteps || ionSteps.length === 0) return null;
            for (var i = 0; i < ionSteps.length; i++) {
                if (ionSteps[i].tof >= targetTime) return ionSteps[i];
            }
            return ionSteps[ionSteps.length - 1];
        }

        function computeAxisRange() {
            const modelData = window.uploadedModelData || {};
            let allY = [], allZ = [];
            for (const m of Object.values(modelData)) {
                if (m.y) allY = allY.concat(Array.from(m.y));
                if (m.z) allZ = allZ.concat(Array.from(m.z));
            }
            if (allY.length === 0) return null;
            const pad = 5;
            const yMin = Math.min(...allY) - pad, yMax = Math.max(...allY) + pad;
            const zMin = Math.min(...allZ) - pad, zMax = Math.max(...allZ) + pad;
            const size = Math.max(yMax - yMin, zMax - zMin);
            const yCtr = (yMin + yMax) / 2, zCtr = (zMin + zMax) / 2;
            return { y: [yCtr - size/2, yCtr + size/2], z: [zCtr - size/2, zCtr + size/2] };
        }

        function getModelPositions(key, time) {
            // Returns {y:[], z:[]} for model at given time (-1 = end positions)
            const modelData = window.uploadedModelData || {};
            const trajData  = window.trajectoryData   || {};
            const m = modelData[key];
            if (!m) return {y:[], z:[]};
            const traj = trajData[key];
            if (traj && time >= 0) {
                const ys = [], zs = [];
                Object.keys(traj).sort((a,b) => parseInt(a)-parseInt(b)).forEach(ionKey => {
                    const pos = getPositionAtTime(traj[ionKey], time);
                    if (pos) { ys.push(pos.y); zs.push(pos.z); }
                });
                return {y: ys, z: zs};
            }
            return {y: Array.from(m.y || []), z: Array.from(m.z || [])};
        }

        function renderUploadedPlot(time) {
            if (time === undefined || time < 0) time = -1;
            const container = document.getElementById('plotly-container');
            const devContainer = document.getElementById('deviation-container');
            const devCell = document.getElementById('deviation-cell');
            const row = document.getElementById('position-plots-row');
            if (!container) return;

            const modelData = window.uploadedModelData || {};
            const isLight   = document.documentElement.getAttribute('data-theme') === 'light';

            if (!window.plotAxisRange && Object.keys(modelData).length > 0) {
                window.plotAxisRange = computeAxisRange();
            }
            const axRange = window.plotAxisRange;

            const squareLayout = {
                xaxis: {
                    title: 'X Position [mm]',
                    gridcolor: isLight ? 'rgba(0,0,0,0.1)' : 'rgba(100,100,120,0.3)',
                    range: axRange ? axRange.y : undefined,
                    fixedrange: true, constrain: 'domain'
                },
                yaxis: {
                    title: 'Y Position [mm]',
                    gridcolor: isLight ? 'rgba(0,0,0,0.1)' : 'rgba(100,100,120,0.3)',
                    range: axRange ? axRange.z : undefined,
                    scaleanchor: 'x', scaleratio: 1,
                    fixedrange: true
                },
                autosize: true,
                height: 480,
                uirevision: 'position-plot',
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: isLight ? 'rgba(250,250,255,0.9)' : 'rgba(25,25,35,0.8)',
                legend: {bgcolor: isLight ? 'rgba(255,255,255,0.9)' : 'rgba(30,30,40,0.9)', font: {color: isLight ? '#333' : '#e0e0e0'}},
                margin: {l:55, r:20, t:20, b:55}
            };

            if (Object.keys(modelData).length === 0) {
                Plotly.react(container, [{x:[], y:[], mode:'markers', type:'scatter', name:'', marker:{size:6}}],
                    Object.assign({}, squareLayout, {annotations:[{text:'Noch kein Modell hochgeladen',
                        xref:'paper', yref:'paper', x:0.5, y:0.5, showarrow:false, font:{size:16, color:'#666'}}]}));
                if (devCell) devCell.style.display = 'none';
                if (row) row.classList.add('single');
                return;
            }

            // --- Main position plot ---
            const checkedKeys = Object.keys(modelData).filter(k => {
                const cb = document.getElementById('cb_' + k);
                return cb && cb.checked;
            });

            const traces = checkedKeys.map(key => {
                const m = modelData[key];
                const pos = getModelPositions(key, time);
                return {x: pos.y, y: pos.z, mode: 'markers', type: 'scatter',
                    name: m.name, marker: {size: 6, color: m.color, opacity: 0.8}};
            });

            Plotly.react(container, traces.length ? traces : [{x:[], y:[], mode:'markers', type:'scatter'}], squareLayout, {responsive: true});

            // --- Deviation plot: only if >=2 checked models and a reference is set ---
            const refKey = window.activeRefKey && modelData[window.activeRefKey] ? window.activeRefKey
                         : (checkedKeys.length > 0 ? checkedKeys[0] : null);

            if (devContainer && devCell && checkedKeys.length >= 2 && refKey) {
                devCell.style.display = 'block';
                if (row) row.classList.remove('single');
                const refPos = getModelPositions(refKey, time);
                const refName = modelData[refKey].name;

                const devTraces = [];
                // Zero-line crosshair
                devTraces.push({x:[-5,5], y:[0,0], mode:'lines', type:'scatter', showlegend:false,
                    line:{color:'rgba(150,150,150,0.4)', width:1, dash:'dash'}, hoverinfo:'skip'});
                devTraces.push({x:[0,0], y:[-5,5], mode:'lines', type:'scatter', showlegend:false,
                    line:{color:'rgba(150,150,150,0.4)', width:1, dash:'dash'}, hoverinfo:'skip'});

                for (const key of checkedKeys) {
                    if (key === refKey) continue;
                    const m = modelData[key];
                    const pos = getModelPositions(key, time);
                    const len = Math.min(pos.y.length, refPos.y.length);
                    const dy = [], dz = [];
                    for (let i = 0; i < len; i++) { dy.push(pos.y[i]-refPos.y[i]); dz.push(pos.z[i]-refPos.z[i]); }
                    devTraces.push({x: dy, y: dz, mode: 'markers', type: 'scatter',
                        name: m.name + ' - ' + refName,
                        marker: {size: 5, color: m.color, opacity: 0.75}});
                }

                const devLayout = {
                    xaxis: {title: 'DeltaX [mm]', gridcolor: isLight ? 'rgba(0,0,0,0.1)' : 'rgba(100,100,120,0.3)',
                        zeroline:false, fixedrange:true, constrain:'domain'},
                    yaxis: {title: 'DeltaY [mm]', gridcolor: isLight ? 'rgba(0,0,0,0.1)' : 'rgba(100,100,120,0.3)',
                        scaleanchor:'x', scaleratio:1, zeroline:false, fixedrange:true},
                    autosize: true, height: 480,
                    uirevision: 'deviation-plot',
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: isLight ? 'rgba(250,250,255,0.9)' : 'rgba(25,25,35,0.8)',
                    legend: {bgcolor: isLight ? 'rgba(255,255,255,0.9)' : 'rgba(30,30,40,0.9)', font: {color: isLight ? '#333' : '#e0e0e0'}},
                    title: {text: 'Deviation from reference: ' + refName, font: {color: isLight ? '#333' : '#e0e0e0', size:13}},
                    margin: {l:55, r:20, t:40, b:55}
                };
                Plotly.react(devContainer, devTraces, devLayout, {responsive: true});
            } else if (devCell) {
                devCell.style.display = 'none';
                if (row) row.classList.add('single');
            }
        }

        function update2DTimePositions(time) {
            renderUploadedPlot(time);
        }

        function updateVisibility() {
            const t = (typeof posCurrentTime !== 'undefined') ? posCurrentTime : -1;
            renderUploadedPlot(t);
        }

        async function loadUploadedModels() {
            const container = document.getElementById('modelSelectionList');

            try {
                const response = await fetch('/get_models');
                const data = await response.json();

                if (!data.uploaded_models || data.uploaded_models.length === 0) {
                    container.innerHTML = '<p style="color: #999; margin: 0; font-size: 11px;">No models uploaded yet...</p>';
                    renderUploadedPlot();  // Show empty axes
                    return;
                }

                // Create checkbox list for each model
                let html = '';
                for (let model of data.uploaded_models) {
                    html += `<div class="model-item" style="margin-bottom: 12px;">
                        <label style="display: flex; align-items: center; gap: 8px; cursor: pointer; color: white; margin: 0;">
                            <input type="checkbox" id="cb_${model.key}" checked onchange="updateVisibility()" style="cursor: pointer;">
                            <span style="color: ${model.color}; font-size: 14px;">&#9679;</span>
                            <span style="flex: 1; font-weight: 500;">${model.name}</span>
                        </label>
                        <div style="font-size: 10px; color: #999; margin-left: 22px; margin-top: 4px;">
                            &#10003; Results${model.has_trajectory ? ' | &#10003; Animations' : ' | &#10060; No Animations'}${model.has_stls ? ' | &#10003; 3D' : ' | &#10060; No 3D'}
                        </div>
                        ${model.has_stls ? `<button onclick="openSTLViewerUploaded('${model.key}', '${model.name}')" style="margin-top: 4px; padding: 4px 8px; background: #667eea; border: none; border-radius: 3px; color: white; cursor: pointer; font-size: 10px;">3D View</button>` : ''}
                    </div>`;
                }

                container.innerHTML = html;

                window.modelKeys = data.uploaded_models.map(m => m.key);
                window.trajectoryData = window.trajectoryData || {};
                window.modelColors = window.modelColors || {};

                // Load full model data (positions + trajectories)
                for (let model of data.uploaded_models) {
                    window.modelColors[model.key] = model.color;
                    try {
                        console.log('Lade Daten für Modell:', model.key);
                        const modelResp = await fetch('/get_model_data/' + model.key);
                        if (modelResp.ok) {
                            const modelData = await modelResp.json();
                            console.log('  Modell-Daten erhalten:', modelData);
                            // Store end positions for plot
                            const endData = (modelData.data || {}).end || {};
                            console.log('  endData:', endData);
                            window.uploadedModelData[model.key] = {
                                y: endData.y || [],
                                z: endData.z || [],
                                color: model.color,
                                name: model.name
                            };
                            console.log('  window.uploadedModelData[' + model.key + '] gespeichert:', window.uploadedModelData[model.key]);
                            if (modelData.trajectories) {
                                window.trajectoryData[model.key] = modelData.trajectories;
                            }
                        } else {
                            console.error('  Fehler: Status', modelResp.status);
                        }
                    } catch(e) {
                        console.warn('Could not load data for', model.key, e);
                    }
                }

                console.log('window.uploadedModelData final:', window.uploadedModelData);

                // Reset fixed axis range so it gets recomputed from new data
                window.plotAxisRange = null;

                // Init scrubber max time from trajectory data (like copy4)
                let posMaxTime = 0;
                for (const key of Object.keys(window.trajectoryData || {})) {
                    for (const ionKey of Object.keys(window.trajectoryData[key])) {
                        const steps = window.trajectoryData[key][ionKey];
                        if (steps && steps.length > 0) {
                            const last = steps[steps.length-1].tof;
                            if (last > posMaxTime) posMaxTime = last;
                        }
                    }
                }
                window.posMaxTime = posMaxTime;
                const scrubber = document.getElementById('posTimeScrubber');
                if (scrubber && posMaxTime > 0) {
                    scrubber.max   = posMaxTime * 1000;
                    scrubber.value = posMaxTime * 1000;
                }
                posCurrentTime = posMaxTime > 0 ? posMaxTime : -1;

                // Show end positions initially (time = -1)
                renderUploadedPlot(-1);

                // Auto-open 3D viewer for first model that has STL data
                const firstWithStl = data.uploaded_models.find(m => m.has_stls);
                if (firstWithStl) {
                    openSTLViewerUploaded(firstWithStl.key, firstWithStl.name);
                }

            } catch (err) {
                container.innerHTML = '<span style="color: #ff6b6b;">Fehler beim Laden: ' + err.message + '</span>';
            }
        }

        // Load uploaded models on page load
        window.addEventListener('load', function() {
            setTimeout(function() {
                renderUploadedPlot(-1);   // Show empty axes immediately
                loadUploadedModels();
            }, 300);
        });
        
        </script>
</body>
</html>
'''
    
    # Write custom HTML without plots (plots will be generated after models are uploaded)
    complete_html = page_header + page_middle + threejs_scripts + js_script
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(complete_html)
    
    # Expose current html filename for optional Flask serving
    try:
        GLOBAL_CURRENT_HTML = str(Path(output_file).resolve().name)
        GLOBAL_HTML_DIR = str(Path(output_file).resolve().parent)
    except Exception:
        GLOBAL_CURRENT_HTML = output_file
        GLOBAL_HTML_DIR = '.'
    
    # Don't open browser automatically - only serve via Flask
    print('\n' + '='*60)
    print('FLASK SERVER STARTED')
    print('='*60)
    print('Oeffne deinen Browser und gehe zu:')
    print('  http://127.0.0.1:5000/')
    print('  oder')
    print('  http://localhost:5000/')
    print('='*60 + '\n')


# Lightweight Flask app for on-demand server-side exports (start with --serve)
flask_app = Flask(__name__)

@flask_app.route('/upload_models', methods=['POST'])
def upload_models_endpoint():
    """
    Handle model file uploads.
    Expects: multipart/form-data with files
    Returns: {success: bool, models: [...], warnings: [...], errors: [...]}
    """
    global UPLOADED_MODELS, NEXT_COLOR_INDEX
    
    try:
        response = {
            'success': False,
            'models': [],
            'warnings': [],
            'errors': []
        }
        
        if 'files' not in request.files:
            response['errors'].append('Keine Dateien hochgeladen')
            return jsonify(response), 400
        
        files = request.files.getlist('files')
        if not files:
            response['errors'].append('Keine Dateien ausgewählt')
            return jsonify(response), 400
        
        model_name = request.form.get('model_name', 'Uploaded_Model')
        model_name = model_name.replace(' ', '_')
        
        # Read uploaded files into memory
        model_files = {}
        for file in files:
            if file.filename:
                try:
                    content = file.read()
                    filename = Path(file.filename).name.lower()
                    model_files[filename] = content if filename.endswith('.stl') else content.decode('utf-8', errors='ignore')
                except Exception as e:
                    response['warnings'].append(f'Datei {file.filename} konnte nicht gelesen werden: {str(e)}')
        
        if not model_files:
            response['errors'].append('Keine Dateien konnten gelesen werden')
            return jsonify(response), 400
        
        # Validate uploaded files
        validation = validate_model_upload(model_files)
        
        if validation['errors']:
            response['errors'].extend(validation['errors'])
            return jsonify(response), 400
        
        response['warnings'].extend(validation['warnings'])
        
        # Parse model data
        parsed_data = parse_uploaded_model(model_name, model_files)
        response['warnings'].extend(parsed_data['warnings'])
        
        # Store in global UPLOADED_MODELS
        color = MODEL_COLORS[NEXT_COLOR_INDEX % len(MODEL_COLORS)]
        NEXT_COLOR_INDEX += 1
        
        model_key = f"uploaded_{len(UPLOADED_MODELS)}_{model_name}"
        UPLOADED_MODELS[model_key] = {
            'name': model_name,
            'color': color,
            'symbol': 'circle',
            'data': {
                'start': parsed_data['start_data'],
                'end': parsed_data['end_data']
            },
            'trajectories': parsed_data['trajectories'],
            'stl_data': parsed_data['stl_data'],
            'has_trajectory': validation['has_trajectory'],
            'has_stls': len(validation['has_stls']) > 0,
            'warnings': parsed_data['warnings']
        }
        
        response['success'] = True
        response['models'].append({
            'key': model_key,
            'name': model_name,
            'color': color,
            'has_stls': validation['has_stls'],
            'has_trajectory': validation['has_trajectory'],
            'ion_count': len(parsed_data['start_data']['ion_n'])
        })
        
        return jsonify(response), 200
    
    except Exception as e:
        import traceback
        print(f"Upload error: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'errors': [f'Server error: {str(e)}']}), 500


@flask_app.route('/get_models', methods=['GET'])
def get_models_endpoint():
    """Get list of available models (built-in + uploaded)"""
    global UPLOADED_MODELS
    
    models_list = []
    
    # Add uploaded models
    for key, model_info in UPLOADED_MODELS.items():
        models_list.append({
            'key': key,
            'name': model_info['name'],
            'color': model_info['color'],
            'source': 'uploaded',
             'has_stls': model_info['has_stls'],
            'has_trajectory': model_info['has_trajectory'],
            'warnings': model_info['warnings']
        })
    
    return jsonify({'uploaded_models': models_list}), 200

@flask_app.route('/get_model_data/<model_key>', methods=['GET'])
def get_model_data(model_key):
    """Get trajectory and position data for a specific model"""
    global UPLOADED_MODELS

    if model_key not in UPLOADED_MODELS:
        return jsonify({'error': 'Model not found'}), 404

    model = UPLOADED_MODELS[model_key]

    # Convert numpy arrays to lists for JSON serialisation
    def to_serialisable(d):
        if isinstance(d, dict):
            return {k: to_serialisable(v) for k, v in d.items()}
        try:
            return d.tolist()
        except AttributeError:
            return d

    return jsonify({
        'key': model_key,
        'name': model['name'],
        'data': to_serialisable(model.get('data', {})),
        'trajectories': model.get('trajectories', {}),
        'has_trajectory': model.get('has_trajectory', False)
    }), 200


@flask_app.route('/get_stl_data/<model_key>', methods=['GET'])
def get_stl_data(model_key):
    """Return STL files for an uploaded model as base64-encoded JSON array (index = electrode number)."""
    global UPLOADED_MODELS

    if model_key not in UPLOADED_MODELS:
        return jsonify({'error': 'Model not found'}), 404

    model = UPLOADED_MODELS[model_key]
    stl_dict = model.get('stl_data', {})

    if not stl_dict:
        return jsonify({'stl_array': [], 'count': 0}), 200

    # Sort by filename so electrode1 < electrode2 < … < electrode9
    import re as _re
    def _electrode_sort_key(name):
        m = _re.search(r'(\d+)', name)
        return int(m.group(1)) if m else 0

    sorted_names = sorted(stl_dict.keys(), key=_electrode_sort_key)

    # Build a dense array (None for missing indices)
    max_idx = _electrode_sort_key(sorted_names[-1]) if sorted_names else 0
    stl_array = [None] * max(max_idx, len(sorted_names))
    for name in sorted_names:
        idx = _electrode_sort_key(name)
        pos = max(idx - 1, 0)
        if pos < len(stl_array):
            data = stl_dict[name]
            # data is already base64 str (encoded in parse_uploaded_model)
            stl_array[pos] = data if isinstance(data, str) else base64.b64encode(data).decode('utf-8')

    return jsonify({'stl_array': stl_array, 'count': len([x for x in stl_array if x])}), 200

@flask_app.route('/export_matplotlib_detector', methods=['POST'])
def export_matplotlib_detector():
    """Export detector positions as matplotlib figure in selected format.
    Expects JSON body: { models: ['key1', ...], format: 'png'|'pdf'|'svg', dpi: 300 }
    Uses UPLOADED_MODELS - no dependency on GLOBAL_ALL_DATA.
    """
    global UPLOADED_MODELS
    try:
        body = request.get_json(force=True, silent=True) or {}
    except Exception:
        body = {}

    requested_keys = body.get('models', [])
    format_type = body.get('format', 'png').lower()
    dpi = int(body.get('dpi', 300))

    if format_type not in ['png', 'pdf', 'svg']:
        return jsonify({'error': 'Invalid format. Must be png, pdf, or svg.'}), 400

    if not UPLOADED_MODELS:
        return jsonify({'error': 'No models uploaded yet. Please upload at least one model first.'}), 400

    # If no specific keys requested, plot all uploaded models
    keys_to_plot = [k for k in requested_keys if k in UPLOADED_MODELS] or list(UPLOADED_MODELS.keys())

    try:
        import io

        # Publication-quality style
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.size': 18,
            'axes.labelsize': 20,
            'axes.labelweight': 'bold',
            'axes.linewidth': 1.8,
            'xtick.labelsize': 17,
            'ytick.labelsize': 17,
            'legend.fontsize': 17,
            'legend.framealpha': 0.95,
        })

        fig = plt.figure(figsize=(7, 7), dpi=dpi)
        ax = fig.add_subplot(111)

        markers = ['o', 's', '^', 'D', 'P', 'X', 'v', '<', '>']
        all_x, all_y = [], []
        for idx, model_key in enumerate(keys_to_plot):
            model_info = UPLOADED_MODELS[model_key]
            end_data = model_info.get('data', {}).get('end', {})
            x_raw = end_data.get('y', [])
            y_raw = end_data.get('z', [])
            x = x_raw.tolist() if hasattr(x_raw, 'tolist') else list(x_raw)
            y = y_raw.tolist() if hasattr(y_raw, 'tolist') else list(y_raw)

            if len(x) == 0:
                continue

            all_x.extend(x)
            all_y.extend(y)

            color = model_info.get('color', '#3498db')
            label = model_info.get('name', model_key)
            marker_style = markers[idx % len(markers)]
            ax.scatter(
                x, y,
                alpha=0.82,
                s=58,
                label=label,
                color=color,
                marker=marker_style,
                edgecolors='white',
                linewidths=0.9
            )
            mean_x = float(sum(x)) / len(x)
            mean_y = float(sum(y)) / len(y)
            if len(x) >= 3:
                coords = np.column_stack((x, y))
                cov = np.cov(coords, rowvar=False)
                if np.all(np.isfinite(cov)):
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    order = eigvals.argsort()[::-1]
                    eigvals = eigvals[order]
                    eigvecs = eigvecs[:, order]
                    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
                    scale = 2.0
                    width = 2.0 * scale * np.sqrt(max(eigvals[0], 0.0))
                    height = 2.0 * scale * np.sqrt(max(eigvals[1], 0.0))
                    ellipse = Ellipse(
                        (mean_x, mean_y),
                        width=width,
                        height=height,
                        angle=angle,
                        fill=False,
                        linestyle='--',
                        linewidth=2.0,
                        edgecolor=color,
                        alpha=0.85,
                        label='outline',
                    )
                    ax.add_patch(ellipse)
            x_min, x_max = min(all_x), max(all_x)
            y_min, y_max = min(all_y), max(all_y)
            span = max(x_max - x_min, y_max - y_min, 1e-6)
            pad = 0.08 * span
            legend_space = 0.40 * span
            ax.set_xlim(x_min - pad, x_max + legend_space)
            ax.set_ylim(y_min - pad, y_max + legend_space)

        ax.set_xlabel('X (mm)', fontsize=20, fontweight='bold')
        ax.set_ylabel('Y (mm)', fontsize=20, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=17, width=1.8, length=6)
        # Reorder legend: pair each model with its outline on the same row
        handles, labels = ax.get_legend_handles_labels()
        scatter_pairs = [(h, l) for h, l in zip(handles, labels) if l != 'outline']
        fit_pairs     = [(h, l) for h, l in zip(handles, labels) if l == 'outline']
        ordered_h, ordered_l = [], []
        # First add all scatter entries, then all outline entries
        # This makes ncol=2 display them row-wise: Model1 outline1 / Model2 outline2
        for h, l in scatter_pairs:
            ordered_h.append(h); ordered_l.append(l)
        for h, l in fit_pairs:
            ordered_h.append(h); ordered_l.append(l)
        ax.legend(ordered_h, ordered_l, frameon=True, fontsize=17, loc='upper right', ncol=2)
        ax.grid(True, alpha=0.25, linestyle=':', linewidth=0.8)
        ax.set_aspect('equal')
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
        fig.tight_layout(pad=1.5)

        buf = io.BytesIO()
        fig.savefig(buf, format=format_type, dpi=dpi, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        mimetype_map = {'png': 'image/png', 'pdf': 'application/pdf', 'svg': 'image/svg+xml'}
        return send_file(buf, mimetype=mimetype_map[format_type], as_attachment=True,
                         download_name=f'vmi_detector_positions.{format_type}')

    except Exception as e:
        print(f'Export error: {e}')
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Export failed: {str(e)}'}), 500

@flask_app.route('/')
def index_root():
    # Redirect to the generated HTML if available
    try:
        return send_from_directory(GLOBAL_HTML_DIR, GLOBAL_CURRENT_HTML)
    except Exception:
        return 'No viewer available. Generate the HTML first.'

# Runs on import too (required for gunicorn, which never calls main())
preload_demo_models()

if __name__ == '__main__':
    # PORT env var lets cloud hosts (Render, Railway, etc.) assign the port dynamically
    port = int(os.environ.get('PORT', 5000))

    # Always start Flask server automatically
    def run_flask():
        print(f'Starting Flask server on port {port}')
        # host='0.0.0.0' allows connections from other devices (hotspot/network) and cloud platforms
        flask_app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
    
    t = threading.Thread(target=run_flask, daemon=True)
    t.start()
    
    main()
    
    print("\n" + "=" * 60)
    print("Web interface ready:")
    print(f"  http://127.0.0.1:{port}")
    print(f"  or http://localhost:{port}")
    print("\nKeep this window open to use the app.")
    print("Press Ctrl+C to stop the server.\n")
    print("=" * 60 + "\n")
    
    # Keep the thread alive
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nServer stopped.")
