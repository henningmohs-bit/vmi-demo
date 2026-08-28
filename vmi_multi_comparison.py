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

import numpy as np
import re
import base64
from pathlib import Path

# Publication export dependencies
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import json

import threading
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

# Demo datasets preloaded on startup (conference setup)
BASE_DIR = Path(__file__).resolve().parent
DEMO_MODEL_FOLDERS = [
    ('Previous design', BASE_DIR / 'Chamber stl' / 'previous', 'demoresult'),
    ('Optimized design', BASE_DIR / 'Chamber stl' / 'optimized', 'demoresult'),
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
        validation['errors'].append('The required "result" file is missing.')
        return validation
    
    validation['has_result'] = True
    validation['valid'] = True
    
    # Check for trajectory data (OPTIONAL)
    # Either a dedicated file OR a large result file (>2MB contains intermediate positions)
    result_content = model_files.get('result', '')
    result_is_large = len(result_content) > 2_000_000
    
    if 'trajectory_data' not in basenames and not result_is_large:
        validation['warnings'].append('No trajectory data found; animation is unavailable.')
    else:
        validation['has_trajectory'] = True
    
    # Check for STL files (OPTIONAL) - robustly search by .stl extension
    stl_files = [f for f in model_files.keys() if f.lower().endswith('.stl')]
    if not stl_files:
        validation['warnings'].append('No STL files found; the 3D view is unavailable.')
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

    for name, folder, result_filename in DEMO_MODEL_FOLDERS:
        try:
            result_path = folder / result_filename
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
        'start_data': {'ion_n': [], 'x': [], 'y': [], 'z': [], 'tof': [], 'ke': []},
        'end_data': {'ion_n': [], 'x': [], 'y': [], 'z': [], 'tof': [], 'ke': []},
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
                result['warnings'].append(f'The result file could not be parsed: {str(e)}')
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
            traj_source_name = 'result (trajectory source)'
        
        if traj_source is not None:
            try:
                result['trajectories'] = parse_trajectory_file_from_content(traj_source)
                print(f"[PARSE] Trajectories from '{traj_source_name}': {len(result['trajectories'])} ions")
            except Exception as e:
                result['warnings'].append(f'Trajectory data could not be parsed: {str(e)}')
        
        # Parse STL files (OPTIONAL)
        for stl_name, stl_content in model_files.items():
            if stl_name.lower().endswith('.stl'):
                try:
                    result['stl_data'][stl_name] = base64.b64encode(stl_content).decode('utf-8')
                except Exception as e:
                    result['warnings'].append(f'STL file {stl_name} could not be encoded: {str(e)}')
    
    except Exception as e:
        result['warnings'].append(f'The model could not be processed: {str(e)}')
    
    return result


def _select_latest_complete_flym_run(content_str):
    """Return the newest complete SIMION Fly'm run from an appended result file."""
    run_marker = re.compile(r"-+\s*Begin Next Fly['’]m\s*-+", re.IGNORECASE)
    runs = [run for run in run_marker.split(content_str) if re.search(r'Ion\(\d+\)\s+Event', run)]
    if not runs:
        return content_str

    # SIMION commonly appends every Fly'm invocation to the same result file and
    # reuses ion numbers in each run. Prefer the newest run whose recorded creation
    # events match the declared ion count; this skips interrupted/partial attempts.
    for run in reversed(runs):
        declared_match = re.search(r'Number of Ions to Fly\s*=\s*(\d+)', run, re.IGNORECASE)
        if not declared_match:
            continue
        declared_count = int(declared_match.group(1))
        created_ions = {
            int(match.group(1))
            for match in re.finditer(
                r'Ion\((\d+)\)\s+Event\([^)]*Created[^)]*\)',
                run,
                re.IGNORECASE,
            )
        }
        if len(created_ions) >= declared_count:
            return run

    # Some valid exports omit the run metadata or creation events. In that case,
    # using only the newest run is still safer than merging reused ion numbers.
    return runs[-1]


def parse_vmi_file_from_content(content_str):
    """Parse start and detector-impact data from the newest complete Fly'm run."""
    start_data = {'ion_n': [], 'x': [], 'y': [], 'z': [], 'tof': [], 'ke': []}
    end_data   = {'ion_n': [], 'x': [], 'y': [], 'z': [], 'tof': [], 'ke': []}
    seen_start_ions = set()
    seen_end_ions   = set()

    if isinstance(content_str, bytes):
        content_str = content_str.decode('utf-8', errors='ignore')

    content_str = _select_latest_complete_flym_run(content_str)
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
        x_match   = re.search(r'X\(([\d.e+-]+)\s*mm\)', block)
        tof_match = re.search(r'TOF\(([\d.e+-]+)\s*usec\)', block)
        ke_match  = re.search(r'KE\(([\d.e+-]+)\s*eV\)', block)

        if not y_match or not z_match:
            continue
        y   = float(y_match.group(1))
        z   = float(z_match.group(1))
        x   = float(x_match.group(1)) if x_match else 0.0
        tof = float(tof_match.group(1)) if tof_match else 0.0
        ke  = float(ke_match.group(1))  if ke_match  else 0.0

        if 'Ion Created' in event_type or 'Created' in event_type:
            if ion_n not in seen_start_ions:
                seen_start_ions.add(ion_n)
                start_data['ion_n'].append(ion_n)
                start_data['x'].append(x)
                start_data['y'].append(y)
                start_data['z'].append(z)
                start_data['tof'].append(tof)
                start_data['ke'].append(ke)
        elif 'Hit Electrode' in event_type or 'Hit' in event_type:
            if ion_n not in seen_end_ions:
                seen_end_ions.add(ion_n)
                end_data['ion_n'].append(ion_n)
                end_data['x'].append(x)
                end_data['y'].append(y)
                end_data['z'].append(z)
                end_data['tof'].append(tof)
                end_data['ke'].append(ke)

    for key in start_data:
        start_data[key] = np.array(start_data[key])
        end_data[key]   = np.array(end_data[key])

    return start_data, end_data


def parse_trajectory_file_from_content(content_str):
    """Parse every ion and timestamp from the newest complete Fly'm run."""
    trajectories = {}

    if isinstance(content_str, bytes):
        content_str = content_str.decode('utf-8', errors='ignore')

    content_str = _select_latest_complete_flym_run(content_str)
    content_str = content_str.replace('\n', ' ')
    ion_blocks  = re.split(r'(?=Ion\(\d+\)\s+Event)', content_str)

    for block in ion_blocks:
        if not block.strip():
            continue
        ion_match = re.search(r'Ion\((\d+)\)', block)
        if not ion_match:
            continue
        ion_n = int(ion_match.group(1))

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
            trajectories.setdefault(ion_n, []).append(timestep)

    # SIMION files normally arrive in chronological order, but sorting each ion's
    # complete series makes the animation and trail rendering deterministic even
    # when uploaded event blocks are out of order.
    for steps in trajectories.values():
        steps.sort(key=lambda step: step['tof'])

    return trajectories


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
    
    # Modern Dark Mode page wrapper with Light/Dark toggle
    page_header = '''<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VMI Simulation Analysis</title>
    <script>
        try {
            var savedVmiTheme = localStorage.getItem('vmi-theme');
            if (savedVmiTheme === 'light' || savedVmiTheme === 'dark') {
                document.documentElement.setAttribute('data-theme', savedVmiTheme);
            }
        } catch (e) {}
    </script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        :root {
            color-scheme: dark;
            --bg-primary: radial-gradient(circle at 12% 0%, #22254a 0%, #111426 38%, #090b14 100%);
            --bg-card: rgba(24, 27, 45, 0.88);
            --bg-card-solid: #181b2d;
            --bg-card-hover: rgba(255,255,255,0.075);
            --bg-input: rgba(255,255,255,0.045);
            --border-color: rgba(255,255,255,0.10);
            --border-strong: rgba(255,255,255,0.17);
            --text-primary: #f3f5fb;
            --text-secondary: #b5bbcc;
            --text-muted: #cbd0dc;
            --stl-bg: #11131e;
            --accent: #6e7ff2;
            --accent-strong: #7654b3;
            --success: #23c990;
            --danger: #f06464;
            --shadow-card: 0 18px 50px rgba(0,0,0,0.22);
            --focus-ring: 0 0 0 4px rgba(110,127,242,0.28);
        }
        
        [data-theme="light"] {
            color-scheme: light;
            --bg-primary: radial-gradient(circle at 12% 0%, #ffffff 0%, #f2f4f9 42%, #e8edf4 100%);
            --bg-card: rgba(255, 255, 255, 0.92);
            --bg-card-solid: #ffffff;
            --bg-card-hover: rgba(45,55,90,0.055);
            --bg-input: rgba(63,72,105,0.045);
            --border-color: rgba(35,43,69,0.11);
            --border-strong: rgba(35,43,69,0.18);
            --text-primary: #22263a;
            --text-secondary: #5f6679;
            --text-muted: #4a5164;
            --stl-bg: #f1f2f7;
            --shadow-card: 0 18px 45px rgba(43,50,77,0.10);
            --focus-ring: 0 0 0 4px rgba(102,126,234,0.20);
        }
        
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: var(--bg-primary);
            background-attachment: fixed;
            min-height: 100vh;
            color: var(--text-primary);
            line-height: 1.5;
            overflow-x: hidden;
            transition: background 0.25s ease, color 0.25s ease;
        }
        body.drawer-open { overflow: hidden; }
        button, input { font: inherit; }
        button { -webkit-tap-highlight-color: transparent; }
        button:focus-visible, input:focus-visible, label:focus-visible {
            outline: none;
            box-shadow: var(--focus-ring);
        }
        .icon {
            width: 18px;
            height: 18px;
            flex: 0 0 auto;
            fill: none;
            stroke: currentColor;
            stroke-width: 1.9;
            stroke-linecap: round;
            stroke-linejoin: round;
        }
        .visually-hidden {
            position: absolute !important;
            width: 1px !important;
            height: 1px !important;
            padding: 0 !important;
            margin: -1px !important;
            overflow: hidden !important;
            clip: rect(0, 0, 0, 0) !important;
            white-space: nowrap !important;
            border: 0 !important;
        }
        .container {
            max-width: 1760px;
            margin: 0 auto;
            padding: 18px clamp(14px, 2.2vw, 32px) 36px;
        }
        header {
            text-align: center;
            padding: 24px 110px 18px;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 14px;
            position: relative;
        }
        header h1 {
            font-size: clamp(1.75rem, 3vw, 2.65rem);
            line-height: 1.1;
            font-weight: 350;
            background: linear-gradient(90deg, #667eea, #764ba2, #d767c9);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 7px;
            letter-spacing: -0.025em;
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
            gap: 8px;
        }
        .theme-label {
            display: grid;
            place-items: center;
            color: var(--text-secondary);
        }
        .toggle-switch {
            position: relative;
            width: 54px;
            height: 28px;
            border-radius: 999px;
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
            height: 22px;
            width: 22px;
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
        .toggle-switch input:focus-visible + .toggle-slider { box-shadow: var(--focus-ring); }
        .toggle-switch input:checked + .toggle-slider:before {
            transform: translateX(26px);
        }
        input[type="range"] { accent-color: var(--accent); }
        
        .main-grid {
            display: block;
        }
        .sidebar {
            position: fixed;
            inset: 0 auto 0 0;
            z-index: 1001;
            width: min(390px, 92vw);
            height: 100dvh;
            display: flex;
            flex-direction: column;
            padding: 0;
            background: var(--bg-card-solid);
            border: 0;
            border-right: 1px solid var(--border-color);
            border-radius: 0 22px 22px 0;
            box-shadow: 24px 0 70px rgba(11, 14, 29, 0.30);
            transform: translateX(-104%);
            visibility: hidden;
            transition: transform 0.28s cubic-bezier(.22,.8,.25,1), visibility 0.28s;
            overflow: hidden;
        }
        .sidebar.is-open {
            transform: translateX(0);
            visibility: visible;
        }
        .drawer-backdrop {
            position: fixed;
            inset: 0;
            z-index: 1000;
            background: rgba(7, 10, 23, 0.48);
            backdrop-filter: blur(4px);
            opacity: 0;
            visibility: hidden;
            transition: opacity 0.24s ease, visibility 0.24s;
        }
        .drawer-backdrop.is-visible {
            opacity: 1;
            visibility: visible;
        }
        .drawer-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            padding: 22px 20px 17px;
            border-bottom: 1px solid var(--border-color);
        }
        .drawer-eyebrow {
            display: block;
            margin-bottom: 2px;
            color: var(--accent);
            font-size: 0.68rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-weight: 750;
        }
        .drawer-header h2 {
            color: var(--text-primary);
            font-size: 1.15rem;
            line-height: 1.25;
            font-weight: 680;
        }
        .drawer-close {
            width: 42px;
            height: 42px;
            display: grid;
            place-items: center;
            flex: 0 0 auto;
            border: 1px solid var(--border-color);
            border-radius: 12px;
            background: var(--bg-input);
            color: var(--text-secondary);
            cursor: pointer;
            transition: background 0.2s ease, color 0.2s ease, transform 0.2s ease;
        }
        .drawer-close:hover {
            background: var(--bg-card-hover);
            color: var(--text-primary);
            transform: translateY(-1px);
        }
        .drawer-body {
            flex: 1;
            min-height: 0;
            overflow-y: auto;
            overscroll-behavior: contain;
            padding: 18px 20px 24px;
        }
        .drawer-footer {
            padding: 14px 20px max(16px, env(safe-area-inset-bottom));
            border-top: 1px solid var(--border-color);
            background: var(--bg-card-solid);
            box-shadow: 0 -12px 35px rgba(11,14,29,0.08);
        }
        /* Conference-ready model drawer */
        .upload-card {
            padding: 14px;
            margin-bottom: 16px;
            border: 1px solid rgba(110,127,242,0.46);
            border-radius: 14px;
            background: linear-gradient(145deg, rgba(110,127,242,0.085), var(--bg-input));
        }
        .section-heading {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 9px;
            color: var(--accent);
            font-size: 0.82rem;
            font-weight: 700;
        }
        .upload-help {
            margin-bottom: 12px;
            color: var(--text-secondary);
            font-size: 0.74rem;
            line-height: 1.45;
        }
        .upload-help strong { color: var(--text-primary); }
        .form-label {
            display: block;
            margin-bottom: 6px;
            color: var(--text-secondary);
            font-size: 0.72rem;
            font-weight: 650;
        }
        .text-input {
            width: 100%;
            min-height: 42px;
            padding: 9px 11px;
            margin-bottom: 10px;
            border: 1px solid var(--border-color);
            border-radius: 10px;
            background: var(--bg-card-solid);
            color: var(--text-primary);
            font-size: 0.78rem;
        }
        .text-input::placeholder { color: var(--text-secondary); opacity: 0.72; }
        .file-picker {
            display: grid;
            grid-template-columns: auto minmax(0, 1fr);
            align-items: center;
            gap: 9px;
            min-height: 44px;
            padding: 5px;
            margin-bottom: 10px;
            border: 1px solid var(--border-color);
            border-radius: 10px;
            background: var(--bg-card-solid);
        }
        .file-picker-button {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
            min-height: 34px;
            padding: 7px 10px;
            border-radius: 7px;
            background: var(--bg-input);
            color: var(--text-primary);
            font-size: 0.72rem;
            font-weight: 650;
            cursor: pointer;
        }
        .file-picker-name {
            min-width: 0;
            overflow: hidden;
            color: var(--text-secondary);
            font-size: 0.7rem;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .btn-primary, .btn-update {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            width: 100%;
            border: 0;
            border-radius: 10px;
            color: #fff;
            cursor: pointer;
            font-weight: 680;
            transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease;
        }
        .btn-primary {
            min-height: 44px;
            padding: 10px 14px;
            background: linear-gradient(135deg, var(--accent), var(--accent-strong));
            box-shadow: 0 8px 20px rgba(102,126,234,0.18);
            font-size: 0.78rem;
        }
        .btn-primary:hover, .btn-update:hover {
            transform: translateY(-1px);
            filter: saturate(1.08);
        }
        .btn-update:hover { box-shadow: 0 8px 25px rgba(56, 239, 125, 0.3); }
        .upload-status {
            display: none;
            align-items: flex-start;
            gap: 8px;
            margin-top: 10px;
            padding: 9px 10px;
            border: 1px solid var(--border-color);
            border-radius: 9px;
            background: var(--bg-input);
            color: var(--text-secondary);
            font-size: 0.73rem;
        }
        .upload-status.is-visible { display: flex; }
        .status-dot {
            width: 8px;
            height: 8px;
            flex: 0 0 auto;
            margin-top: 5px;
            border-radius: 999px;
            background: var(--accent);
            box-shadow: 0 0 0 4px rgba(110,127,242,0.13);
        }
        .upload-status[data-type="success"] .status-dot { background: var(--success); box-shadow: 0 0 0 4px rgba(35,201,144,0.13); }
        .upload-status[data-type="error"] .status-dot { background: var(--danger); box-shadow: 0 0 0 4px rgba(240,100,100,0.13); }
        .upload-status[data-type="loading"] .status-dot { animation: status-pulse 1s ease-in-out infinite; }
        @keyframes status-pulse { 50% { opacity: 0.35; transform: scale(0.82); } }
        .model-list {
            display: grid;
            gap: 9px;
            min-height: 48px;
            padding: 0;
            border: 0;
            background: transparent;
        }
        .empty-state {
            padding: 16px 10px;
            border: 1px dashed var(--border-strong);
            border-radius: 12px;
            color: var(--text-secondary);
            font-size: 0.75rem;
            text-align: center;
        }
        .model-item {
            display: grid;
            align-items: stretch;
            gap: 9px;
            padding: 12px;
            margin: 0;
            border: 1px solid var(--border-color);
            border-radius: 12px;
            background: var(--bg-input);
            transition: background 0.2s ease, border-color 0.2s ease, transform 0.2s ease;
        }
        .model-item:hover {
            background: var(--bg-card-hover);
            border-color: var(--border-strong);
            transform: translateY(-1px);
        }
        .model-item-main {
            display: flex;
            align-items: center;
            gap: 9px;
        }
        .model-toggle {
            display: flex;
            align-items: center;
            gap: 9px;
            min-width: 0;
            flex: 1;
            color: var(--text-primary);
            font-weight: 500;
            cursor: pointer;
        }
        .model-item input[type="checkbox"] {
            width: 19px;
            height: 19px;
            flex: 0 0 auto;
            margin: 0;
            accent-color: var(--accent);
            cursor: pointer;
        }
        .model-color {
            width: 10px;
            height: 10px;
            flex: 0 0 auto;
            border-radius: 999px;
            box-shadow: 0 0 0 3px var(--bg-card-solid);
        }
        .model-name {
            min-width: 0;
            overflow: hidden;
            color: var(--text-primary);
            font-size: 0.85rem;
            font-weight: 680;
            text-overflow: ellipsis;
        }
        .model-item .btn-3d {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 5px;
            min-height: 34px;
            padding: 6px 9px;
            border: 0;
            border-radius: 8px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: #fff;
            cursor: pointer;
            font-size: 0.68rem;
            font-weight: 650;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .model-item .btn-3d:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(102,126,234,0.28);
        }
        .model-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            padding-left: 28px;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 3px 7px;
            border-radius: 999px;
            background: rgba(35,201,144,0.10);
            color: var(--success);
            font-size: 0.62rem;
            font-weight: 700;
            line-height: 1.35;
        }
        .status-pill::before {
            content: '';
            width: 5px;
            height: 5px;
            border-radius: 999px;
            background: currentColor;
        }
        .status-pill.is-missing {
            background: rgba(240,100,100,0.09);
            color: var(--danger);
        }
        .btn-update {
            min-height: 46px;
            padding: 12px 14px;
            margin: 0;
            background: linear-gradient(135deg, #11998e, #38d982);
            box-shadow: 0 8px 22px rgba(35,201,144,0.20);
            font-size: 0.8rem;
            text-transform: none;
            letter-spacing: 0;
        }
        .content-area {
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .plot-container,
        .stl-viewer-section {
            background: var(--bg-card);
            border-radius: 16px;
            padding: 20px;
            border: 1px solid var(--border-color);
            box-shadow: var(--shadow-card);
            backdrop-filter: blur(12px);
        }
        #plotly-container {
            width: 100%;
            height: 480px;
        }
        .stl-viewer-section {
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
            gap: 14px;
            margin-bottom: 15px;
        }
        .stl-viewer-header h3 {
            font-size: 1.1em;
            font-weight: 500;
            color: var(--text-muted);
        }
        .btn-close-3d {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
            min-height: 40px;
            padding: 8px 13px;
            background: rgba(240, 100, 100, 0.12);
            border: 1px solid rgba(240, 100, 100, 0.28);
            border-radius: 8px;
            color: var(--danger);
            cursor: pointer;
            font-weight: 650;
            transition: all 0.2s ease;
        }
        .btn-close-3d:hover {
            background: var(--danger);
            color: #fff;
        }
        #stlContainer {
            width: 100%;
            height: 420px;
            border-radius: 10px;
            overflow: hidden;
            background: var(--stl-bg);
            border: 1px solid var(--border-color);
        }
        #stlContainer canvas { display: block; width: 100%; height: 100%; touch-action: none; }
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

        /* Header actions */
        .header-toolbar {
            position: sticky;
            top: 10px;
            z-index: 90;
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 7px;
            margin: 0 0 18px;
            border: 1px solid var(--border-color);
            border-radius: 14px;
            background: var(--bg-card);
            box-shadow: 0 10px 32px rgba(28,34,59,0.10);
            backdrop-filter: blur(16px);
        }
        .toolbar-spacer { flex: 1; }
        .toolbar-button {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            min-height: 42px;
            padding: 9px 13px;
            border: 1px solid transparent;
            border-radius: 10px;
            background: transparent;
            color: var(--text-secondary);
            cursor: pointer;
            font-size: 0.78rem;
            font-weight: 650;
            transition: background 0.2s ease, color 0.2s ease, border-color 0.2s ease, transform 0.2s ease;
        }
        .toolbar-button:hover {
            background: var(--bg-card-hover);
            border-color: var(--border-color);
            color: var(--text-primary);
            transform: translateY(-1px);
        }
        .toolbar-button.is-primary {
            background: linear-gradient(135deg, var(--accent), var(--accent-strong));
            color: #fff;
            box-shadow: 0 7px 18px rgba(102,126,234,0.20);
        }
        .model-count {
            min-width: 25px;
            padding: 2px 6px;
            border-radius: 999px;
            background: rgba(255,255,255,0.18);
            color: inherit;
            font-size: 0.66rem;
            font-variant-numeric: tabular-nums;
            text-align: center;
        }

        .control-button {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 7px;
            min-height: 40px;
            padding: 8px 14px;
            border: 0;
            border-radius: 8px;
            color: #fff;
            cursor: pointer;
            font-size: 0.78rem;
            font-weight: 650;
        }
        .control-button.is-play { background: linear-gradient(135deg, var(--accent), var(--accent-strong)); }
        .control-button.is-reset { background: var(--danger); }
        .control-button:disabled { opacity: 0.48; cursor: not-allowed; filter: grayscale(0.25); }
        .control-button .icon-pause { display: none; }
        .control-button.is-playing .icon-play { display: none; }
        .control-button.is-playing .icon-pause { display: block; }
        .time-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
            min-width: 108px;
            padding: 7px 10px;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            background: var(--bg-input);
            color: var(--text-secondary);
            font-size: 0.75rem;
            font-variant-numeric: tabular-nums;
        }
        .time-badge strong { color: var(--text-primary); font-weight: 700; }
        .stl-option {
            display: flex;
            align-items: center;
            gap: 7px;
            color: var(--text-secondary);
            font-size: 0.76rem;
            cursor: pointer;
        }
        .stl-option input { width: 17px; height: 17px; accent-color: var(--accent); }
        .stl-header-actions { display: flex; align-items: center; gap: 12px; }
        .stl-overlay { box-shadow: 0 10px 30px rgba(10,12,25,0.14); border: 1px solid var(--border-color); }
        #clipControls button { min-height: 32px; }
        #clipControls input[type="range"], #trajControls input[type="range"], #posTimeScrubber { min-height: 24px; }

        .toast-region {
            position: fixed;
            right: 18px;
            bottom: 18px;
            z-index: 1200;
            display: grid;
            gap: 8px;
            width: min(360px, calc(100vw - 28px));
            pointer-events: none;
        }
        .toast {
            padding: 12px 14px;
            border: 1px solid var(--border-color);
            border-left: 4px solid var(--accent);
            border-radius: 11px;
            background: var(--bg-card-solid);
            color: var(--text-primary);
            box-shadow: var(--shadow-card);
            font-size: 0.78rem;
            animation: toast-in 0.22s ease both;
        }
        .toast.is-error { border-left-color: var(--danger); }
        .toast.is-success { border-left-color: var(--success); }
        @keyframes toast-in { from { opacity: 0; transform: translateY(8px); } }
        
        /* Visualization controls and tabs */
        .tab-bar,
        .main-tab-bar {
            display: flex;
            gap: 0;
            background: var(--bg-input);
            border-radius: 12px;
            padding: 4px;
            border: 1px solid var(--border-color);
        }
        .tab-bar { margin-bottom: 15px; }
        .main-tab-bar { margin-bottom: 20px; }
        .tab-btn,
        .main-tab-btn {
            flex: 1;
            border: none;
            background: transparent;
            color: var(--text-secondary);
            font-weight: 500;
            cursor: pointer;
            border-radius: 8px;
            transition: all 0.3s ease;
        }
        .tab-btn {
            padding: 12px 20px;
            font-size: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }
        .main-tab-btn {
            padding: 14px 16px;
            font-size: 13px;
            text-align: center;
        }
        .tab-btn:hover,
        .main-tab-btn:hover {
            background: var(--bg-card-hover);
            color: var(--text-primary);
        }
        .main-tab-btn.active {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        .main-tab-panel.hidden {
            display: none;
        }
        .main-tab-panel.active {
            display: block !important;
        }
        /* ===== Mobile / Tablet Responsiveness ===== */
        @media (max-width: 900px) {
            .header-toolbar { top: 6px; }
            .stl-viewer-header { align-items: flex-start; flex-wrap: wrap; }
            .stl-header-actions { width: 100%; justify-content: space-between; }
        }
        @media (max-width: 768px) {
            body { background-attachment: scroll; }
            .container {
                padding: 8px 10px 24px;
            }
            header {
                padding: 58px 6px 16px;
                margin-bottom: 10px;
            }
            header h1 {
                font-size: clamp(1.45rem, 7vw, 1.9rem);
            }
            header p {
                font-size: 0.82rem;
            }
            .theme-toggle {
                top: 10px;
                right: 4px;
                transform: none;
            }
            .header-toolbar {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 6px;
                padding: 6px;
                margin-bottom: 12px;
                border-radius: 12px;
            }
            .toolbar-button {
                width: 100%;
                min-height: 44px;
                padding: 8px 9px;
                font-size: 0.72rem;
            }
            #modelDrawerToggle { grid-column: 1 / -1; }
            .toolbar-spacer { display: none; }
            .sidebar {
                width: min(380px, 94vw);
                border-radius: 0 18px 18px 0;
            }
            .drawer-header { padding: 17px 15px 14px; }
            .drawer-body { padding: 14px 15px 20px; }
            .drawer-footer { padding-left: 15px; padding-right: 15px; }
            .content-area { gap: 12px; }
            .plot-container, .stl-viewer-section {
                padding: 11px;
                border-radius: 13px;
            }
            .stl-viewer-header h3 { font-size: 0.95rem; }
            .stl-header-actions { gap: 8px; }
            .stl-option { font-size: 0.7rem; }
            .btn-close-3d { min-height: 38px; padding: 7px 10px; font-size: 0.72rem; }
            #clipControls, #trajControls { padding: 10px !important; }
            #trajControls > div:first-child { gap: 9px !important; }
            #trajControls > div:first-child > div { min-width: 100% !important; order: 3; }
            #trajControls > div:first-child > label { margin-left: auto; }
            #trajIonCount { margin-left: 0 !important; width: 100%; }
            .control-button { min-height: 42px; }
            .stl-overlay {
                max-height: 72%;
                overflow-y: auto;
                transform: scale(0.86);
                transform-origin: top right;
            }
            #stlContainer {
                height: clamp(280px, 54vh, 390px);
            }
            #plotly-container {
                height: 400px;
            }
            .main-tab-bar {
                overflow-x: auto;
                -webkit-overflow-scrolling: touch;
                margin-bottom: 12px;
            }
            .main-tab-btn {
                flex: 0 0 auto;
                padding: 10px 14px;
                font-size: 12px;
            }
            .tab-bar {
                gap: 8px !important;
                padding: 7px !important;
            }
            .tab-bar > span:first-child { width: 100%; }
            #posTimeScrubber { order: 5; flex-basis: 100% !important; }
            .time-badge { margin-left: auto; }
            .toast-region { right: 10px; bottom: 10px; }
        }
        @media (max-width: 430px) {
            .toolbar-button span.toolbar-label { white-space: normal; }
            .stl-header-actions { align-items: center; }
            .stl-option { max-width: 165px; }
            .model-meta { padding-left: 0; }
            .file-picker { grid-template-columns: 1fr; }
            .file-picker-button { width: 100%; }
            .file-picker-name { padding: 0 5px 4px; text-align: center; }
        }
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                scroll-behavior: auto !important;
                animation-duration: 0.01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: 0.01ms !important;
            }
        }
    </style>
</head>
<body class="light-mode">
    <div class="container">
        <header>
            <h1 id="mainTitle">VMI Simulation Analysis</h1>
            <p id="mainSubtitle">Interactive model comparison and ion-trajectory analysis</p>
            <div class="theme-toggle">
                <span class="theme-label" title="Dark theme" aria-hidden="true">
                    <svg class="icon" viewBox="0 0 24 24"><path d="M20.5 14.2A8.2 8.2 0 0 1 9.8 3.5 8.7 8.7 0 1 0 20.5 14.2Z"/></svg>
                </span>
                <label class="toggle-switch">
                    <input type="checkbox" id="themeToggle" onchange="toggleTheme(true)" aria-label="Use light theme" checked>
                    <span class="toggle-slider"></span>
                </label>
                <span class="theme-label" title="Light theme" aria-hidden="true">
                    <svg class="icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="3.5"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>
                </span>
            </div>
        </header>        
        
        <div class="header-toolbar" aria-label="Viewer actions">
            <button type="button" class="toolbar-button is-primary" id="modelDrawerToggle" onclick="toggleModelDrawer()" aria-controls="modelDrawer" aria-expanded="false">
                <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M4 6h16M4 12h16M4 18h16"/></svg>
                <span class="toolbar-label">Models</span>
                <span class="model-count" id="modelCountBadge" aria-label="No models loaded">0</span>
            </button>
            <span class="toolbar-spacer" aria-hidden="true"></span>
            <button type="button" class="toolbar-button" onclick="downloadDetectorPositions('png')">
                <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3v12M7.5 10.5 12 15l4.5-4.5M4 20h16"/></svg>
                <span class="toolbar-label">Export detector plot</span>
            </button>
            <button type="button" class="toolbar-button" onclick="download3D()">
                <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="m12 3 8 4.5v9L12 21l-8-4.5v-9L12 3Z M4.4 7.7 12 12l7.6-4.3M12 12v9"/></svg>
                <span class="toolbar-label">Export 3D view</span>
            </button>
        </div>

        <div class="drawer-backdrop" id="modelDrawerBackdrop" onclick="closeModelDrawer()" aria-hidden="true"></div>

        <div class="main-grid">
            <aside class="sidebar" id="modelDrawer" aria-hidden="true" aria-labelledby="modelDrawerTitle">
                <div class="drawer-header">
                    <div>
                        <span class="drawer-eyebrow">Comparison setup</span>
                        <h2 id="modelDrawerTitle">Model selection</h2>
                    </div>
                    <button type="button" class="drawer-close" onclick="closeModelDrawer()" aria-label="Close model selection">
                        <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="m6 6 12 12M18 6 6 18"/></svg>
                    </button>
                </div>

                <div class="drawer-body">
                    <!-- MODEL UPLOAD SECTION -->
                    <section class="upload-card" aria-labelledby="uploadHeading">
                        <h3 class="section-heading" id="uploadHeading">
                            <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M3 7.5h6l2-2h10v13H3v-11Z"/><path d="M12 16V9M9.5 11.5 12 9l2.5 2.5"/></svg>
                            Load model folder
                        </h3>
                        <p class="upload-help">Choose a folder containing a <strong>result</strong> file. Trajectory data and <strong>electrode*.stl</strong> files are optional.</p>

                        <label class="form-label" for="modelNameInput">Model name</label>
                        <input class="text-input" type="text" id="modelNameInput" placeholder="Detected from folder name">

                        <div class="file-picker">
                            <input class="visually-hidden" type="file" id="modelFolderInput" webkitdirectory directory mozdirectory>
                            <label class="file-picker-button" for="modelFolderInput" tabindex="0">
                                <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M3 7.5h6l2-2h10v13H3v-11Z"/></svg>
                                Choose folder
                            </label>
                            <span class="file-picker-name" id="modelFolderLabel">No folder selected</span>
                        </div>

                        <button type="button" class="btn-primary" onclick="handleModelUpload()">
                            <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 16V4M7.5 8.5 12 4l4.5 4.5M4 20h16"/></svg>
                            Load folder
                        </button>
                        <div id="uploadStatus" class="upload-status" role="status" aria-live="polite">
                            <span class="status-dot" aria-hidden="true"></span>
                            <div><span id="uploadStatusMessage"></span><div id="uploadStatusDetails"></div></div>
                        </div>
                    </section>

                    <!-- UPLOADED MODELS LIST -->
                    <div id="modelSelectionList" class="model-list">
                        <p class="empty-state">No models loaded yet.</p>
                    </div>
                </div>

                <div class="drawer-footer">
                    <button type="button" class="btn-update" onclick="applyModelSelection()">
                        <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="m5 12.5 4.2 4.2L19 7"/></svg>
                        Apply selection
                    </button>
                </div>
            </aside>

            
            <div class="content-area">
                <div class="stl-viewer-section active" id="stlSection">
                    <div class="stl-viewer-header">
                        <h3 id="stlTitle">3D Model Viewer</h3>
                        <div class="stl-header-actions">
                            <label class="stl-option">
                                <input type="checkbox" id="trajShowTrajectories" checked onchange="toggleTrajectories()"> Show Trajectories
                            </label>
                            <button type="button" class="btn-close-3d" onclick="closeSTLViewer()">
                                <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="m6 6 12 12M18 6 6 18"/></svg>
                                Close
                            </button>
                        </div>
                    </div>
                    <div id="stlContainer" style="display: flex; align-items: center; justify-content: center; color: var(--text-secondary); font-size: 13px;">
                        No model loaded &mdash; upload a folder with STL files to see the 3D view
                    </div>
                    
                    <!-- Clipping Plane Controls -->
                    <div id="clipControls" style="margin-top: 12px; padding: 12px; background: var(--bg-input); border-radius: 10px; border: 1px solid var(--border-color);">
                        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                            <strong style="color: var(--text-primary); font-size: 12px;">Section Planes</strong>
                            <button onclick="toggleGridVisibility()" style="padding: 4px 8px; background: rgba(100,100,100,0.3); border: 1px solid var(--border-color); border-radius: 4px; color: var(--text-secondary); cursor: pointer; font-size: 10px;">Grid</button>
                            <button onclick="resetClipping()" style="margin-left: auto; padding: 4px 10px; background: rgba(100,100,100,0.3); border: 1px solid var(--border-color); border-radius: 4px; color: var(--text-secondary); cursor: pointer; font-size: 10px;">Reset</button>
                        </div>
                        <div style="display: grid; grid-template-columns: 50px 1fr 50px; gap: 6px; align-items: center;">
                            <span style="color: #ef5350; font-size: 11px; font-weight: 500;">X:</span>
                            <input type="range" id="clipX" min="-100" max="100" value="0" style="width: 100%; cursor: pointer;" oninput="updateClipping()">
                            <span id="clipXLabel" style="color: var(--text-secondary); font-size: 10px;">0mm</span>
                            
                            <span style="color: #66bb6a; font-size: 11px; font-weight: 500;">Y:</span>
                            <input type="range" id="clipY" min="-100" max="100" value="100" style="width: 100%; cursor: pointer;" oninput="updateClipping()">
                            <span id="clipYLabel" style="color: var(--text-secondary); font-size: 10px;">Off</span>
                            
                            <span style="color: #42a5f5; font-size: 11px; font-weight: 500;">Z:</span>
                            <input type="range" id="clipZ" min="-100" max="100" value="100" style="width: 100%; cursor: pointer;" oninput="updateClipping()">
                            <span id="clipZLabel" style="color: var(--text-secondary); font-size: 10px;">Off</span>
                        </div>
                    </div>
                    
                    <!-- Trajectory Animation Controls -->
                    <div id="trajControls" style="margin-top: 12px; padding: 12px; background: var(--bg-input); border-radius: 10px; border: 1px solid var(--border-color);">
                        <div style="display: flex; align-items: center; gap: 15px; flex-wrap: wrap;">
                            <button type="button" id="trajPlayBtn" class="control-button is-play" onclick="toggleTrajectoryPlay()">
                                <svg class="icon icon-play" viewBox="0 0 24 24" aria-hidden="true"><path d="m8 5 11 7-11 7V5Z"/></svg>
                                <svg class="icon icon-pause" viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14M16 5v14"/></svg>
                                <span id="trajPlayLabel">Play</span>
                            </button>
                            <button type="button" class="control-button is-reset" onclick="resetTrajectory()">
                                <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M4 12a8 8 0 1 0 2.3-5.7L4 8.6M4 4v4.6h4.6"/></svg>
                                Reset
                            </button>
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
                <div class="main-tab-bar" id="mainTabBar">
                    <button type="button" class="main-tab-btn active" id="main-tab-positions" onclick="switchMainTab('positions')">Positions</button>
                        <button type="button" class="main-tab-btn" id="main-tab-tof" onclick="switchMainTab('tof')" style="display:none;">Time of Flight</button>
                        <button type="button" class="main-tab-btn" id="main-tab-radial" onclick="switchMainTab('radial')" style="display:none;">Radial Distribution</button>
                        <button type="button" class="main-tab-btn" id="main-tab-energy" onclick="switchMainTab('energy')" style="display:none;">Kinetic Energy</button>
                </div>
                
                <!-- Positions Panel -->
                <div class="main-tab-panel active" id="panel-positions">
                    <div class="plot-container">
                        <div class="tab-bar" style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
                            <span style="font-size: 12px; color: var(--text-secondary); font-weight: 600;">Flight time</span>
                            <button type="button" class="tab-btn" id="btnJumpStart" onclick="jumpToTime(0)" style="padding: 7px 11px;">
                                <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M6 5v14M18 6l-8 6 8 6V6Z"/></svg>
                                Start
                            </button>
                            <input type="range" id="posTimeScrubber" min="0" max="1000" value="0" 
                                   style="flex: 1; min-width: 150px; cursor: pointer;" 
                                   oninput="scrubPositionTime()">
                            <button type="button" class="tab-btn" id="btnJumpEnd" onclick="jumpToTime(-1)" style="padding: 7px 11px;">
                                End
                                <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M18 5v14M6 6l8 6-8 6V6Z"/></svg>
                            </button>
                            <span id="posTimeLabel" class="time-badge">
                                <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="8.5"/><path d="M12 7v5l3.5 2"/></svg>
                                <strong id="posTimeValue">0.000 &micro;s</strong>
                            </span>
                        </div>
                        <div id="plotly-container"></div>
'''
    
    page_middle = '''
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
    <div class="toast-region" id="toastRegion" aria-live="polite" aria-atomic="true"></div>
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
    model_charges_js = "var modelCharges = "
    model_charges_js += json.dumps(model_charges) + ";\n"
    # Embed statistics (computed earlier) for client-side downloads &mdash; do not auto-generate files on run
    try:
        stats_js = "var allStats = " + json.dumps(all_stats) + ";\n"
    except Exception:
        stats_js = "var allStats = {};\n"
    
    # Generate trajectory data as JavaScript
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
    var lastDrawerFocus = null;

    function setModelDrawerOpen(open) {
        var drawer = document.getElementById('modelDrawer');
        var backdrop = document.getElementById('modelDrawerBackdrop');
        var toggle = document.getElementById('modelDrawerToggle');
        if (!drawer || !backdrop || !toggle) return;

        if (open) {
            lastDrawerFocus = document.activeElement;
            drawer.classList.add('is-open');
            backdrop.classList.add('is-visible');
            drawer.setAttribute('aria-hidden', 'false');
            toggle.setAttribute('aria-expanded', 'true');
            drawer.inert = false;
            document.body.classList.add('drawer-open');
            window.setTimeout(function() {
                var closeButton = drawer.querySelector('.drawer-close');
                if (closeButton) closeButton.focus();
            }, 60);
        } else {
            drawer.classList.remove('is-open');
            backdrop.classList.remove('is-visible');
            drawer.setAttribute('aria-hidden', 'true');
            toggle.setAttribute('aria-expanded', 'false');
            drawer.inert = true;
            document.body.classList.remove('drawer-open');
            if (lastDrawerFocus && typeof lastDrawerFocus.focus === 'function') {
                lastDrawerFocus.focus();
            }
        }
    }

    function toggleModelDrawer() {
        var drawer = document.getElementById('modelDrawer');
        setModelDrawerOpen(!(drawer && drawer.classList.contains('is-open')));
    }

    function closeModelDrawer() {
        setModelDrawerOpen(false);
    }

    function applyModelSelection() {
        updateVisibility();
        closeModelDrawer();
    }

    function updateModelSelectionSummary() {
        var list = document.getElementById('modelSelectionList');
        var badge = document.getElementById('modelCountBadge');
        if (!list || !badge) return;
        var boxes = Array.from(list.querySelectorAll('input[type="checkbox"]'));
        var selected = boxes.filter(function(box) { return box.checked; }).length;
        badge.textContent = boxes.length ? selected + '/' + boxes.length : '0';
        badge.setAttribute('aria-label', boxes.length
            ? selected + ' of ' + boxes.length + ' models selected'
            : 'No models loaded');
    }

    function showToast(message, type) {
        var region = document.getElementById('toastRegion');
        if (!region) return;
        var toast = document.createElement('div');
        toast.className = 'toast' + (type ? ' is-' + type : '');
        toast.setAttribute('role', type === 'error' ? 'alert' : 'status');
        toast.textContent = message;
        region.appendChild(toast);
        window.setTimeout(function() {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(6px)';
            window.setTimeout(function() { toast.remove(); }, 220);
        }, 3600);
    }

    function setUploadStatus(type, message, details) {
        var status = document.getElementById('uploadStatus');
        var messageEl = document.getElementById('uploadStatusMessage');
        var detailsEl = document.getElementById('uploadStatusDetails');
        if (!status || !messageEl || !detailsEl) return;
        status.dataset.type = type || 'info';
        status.classList.add('is-visible');
        messageEl.textContent = message || '';
        detailsEl.replaceChildren();
        if (Array.isArray(details) && details.length) {
            var list = document.createElement('ul');
            list.style.cssText = 'margin: 5px 0 0 16px; color: var(--text-secondary);';
            details.forEach(function(detail) {
                var item = document.createElement('li');
                item.textContent = detail;
                list.appendChild(item);
            });
            detailsEl.appendChild(list);
        }
    }

    function setTrajectoryPlayButton(isPlaying) {
        var button = document.getElementById('trajPlayBtn');
        var label = document.getElementById('trajPlayLabel');
        if (!button || !label) return;
        button.classList.toggle('is-playing', Boolean(isPlaying));
        label.textContent = isPlaying ? 'Pause' : 'Play';
        button.setAttribute('aria-label', isPlaying ? 'Pause trajectory animation' : 'Play trajectory animation');
    }

    function updateMainTabBarVisibility() {
        var bar = document.getElementById('mainTabBar');
        if (!bar) return;
        var buttons = Array.from(bar.querySelectorAll('.main-tab-btn'));
        var visibleButtons = buttons.filter(function(button) {
            return window.getComputedStyle(button).display !== 'none';
        });
        bar.hidden = visibleButtons.length <= 1;
    }

    function resizeSTLViewer() {
        var container = document.getElementById('stlContainer');
        if (!container || !stlRenderer || !stlCamera || !container.clientWidth || !container.clientHeight) return;
        stlCamera.aspect = container.clientWidth / container.clientHeight;
        stlCamera.updateProjectionMatrix();
        stlRenderer.setSize(container.clientWidth, container.clientHeight, false);
    }

    document.addEventListener('DOMContentLoaded', function() {
        var drawer = document.getElementById('modelDrawer');
        if (drawer) drawer.inert = true;
        updateMainTabBarVisibility();

        var savedTheme = null;
        try { savedTheme = localStorage.getItem('vmi-theme'); } catch (e) {}
        var themeToggle = document.getElementById('themeToggle');
        if (themeToggle && (savedTheme === 'light' || savedTheme === 'dark')) {
            themeToggle.checked = savedTheme === 'light';
        }
        toggleTheme(false);

        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape') closeModelDrawer();
        });
        window.addEventListener('resize', resizeSTLViewer);
    });
    
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
        // Conference KOS: Z axis vertical, X axis points into the scene.
        // Default to a frontal view straight onto the YZ plane (looking along -X).
        stlCamera.up.set(0, 0, 1);
        stlCamera.position.set(220, 0, 0);
        
        // Enable preserveDrawingBuffer so canvas.toDataURL works for image export
        stlRenderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true, alpha: false });
        stlRenderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        stlRenderer.setSize(container.clientWidth, container.clientHeight);
        // Set clear color to match theme background
        stlRenderer.setClearColor(isLightTheme ? 0xf0f0f5 : 0x15151f, 1);
        stlRenderer.localClippingEnabled = true;
        container.appendChild(stlRenderer.domElement);
        if (window.ResizeObserver) {
            if (window.stlResizeObserver) window.stlResizeObserver.disconnect();
            window.stlResizeObserver = new ResizeObserver(function() { resizeSTLViewer(); });
            window.stlResizeObserver.observe(container);
        }
        
        // Initialize clipping planes (pointing inward, initially disabled at max values)
        clipPlaneX = new THREE.Plane(new THREE.Vector3(-1, 0, 0), 250);
        clipPlaneY = new THREE.Plane(new THREE.Vector3(0, -1, 0), 100);
        clipPlaneZ = new THREE.Plane(new THREE.Vector3(0, 0, -1), 100);
        
        // Reset clipping sliders (percentage-based; actual mm ranges set once bounding box is known)
        document.getElementById('clipX').value = 0;
        document.getElementById('clipY').value = 100;
        document.getElementById('clipZ').value = 100;
        document.getElementById('clipXLabel').textContent = '0mm';
        document.getElementById('clipYLabel').textContent = 'Off';
        document.getElementById('clipZLabel').textContent = 'Off';
        
        // Initialize clipping planes at the default scene state
        if (clipPlaneX) clipPlaneX.constant = 0;
        if (clipPlaneY) clipPlaneY.constant = 0;
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
        
        // Add grid (hidden by default; can be toggled for reference)
        var gridHelper = new THREE.GridHelper(400, 40);
        gridHelper.visible = false;
        stlScene.add(gridHelper);
        window.stlGridHelper = gridHelper;
        
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
        // Reference bounding box used to align trajectory data - excludes electrode7 (index 6),
        // since its geometry doesn't reliably represent the chamber's true center/edges.
        var refBoundingBox = new THREE.Box3();
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
                if (i !== 6) {
                    refBoundingBox.union(geometry.boundingBox);
                }
                
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
            
            // Store the electrode7-excluded reference bbox in the same centered coordinate
            // frame as the meshes, so trajectory alignment can use it directly.
            if (!refBoundingBox.isEmpty()) {
                var refCenter = new THREE.Vector3();
                refBoundingBox.getCenter(refCenter);
                window.trajAlignRef = {
                    centerX: refCenter.x - center.x,
                    centerZ: refCenter.z - center.z,
                    maxY: refBoundingBox.max.y - center.y
                };
            } else {
                window.trajAlignRef = { centerX: 0, centerZ: 0, maxY: 0 };
            }
            
            // Auto-scale camera based on overall size
            var size = new THREE.Vector3();
            globalBoundingBox.getSize(size);
            var maxDim = Math.max(size.x, size.y, size.z);
            
            // Store per-axis half-extents (with padding) so the Section Plane sliders
            // work as a percentage of the actual model size, not a fixed mm range.
            window.clipRanges = {
                x: (size.x / 2) * 1.1 || 1,
                y: (size.y / 2) * 1.1 || 1,
                z: (size.z / 2) * 1.1 || 1
            };
            updateClipping();
            
            // Frontal view straight onto the YZ plane: camera sits on the X axis
            // looking along -X, so Y runs left/right and Z runs bottom/top with no tilt.
            stlCamera.position.set(maxDim * 2.2, 0, 0);
            stlCamera.up.set(0, 0, 1);
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
        legendDiv.className = 'stl-overlay stl-legend';
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
                legendItem.innerHTML = '<span class="legend-color" style="width: 10px; height: 10px; border-radius: 2px; background: ' + colorHex + '; flex: 0 0 auto;"></span><span class="legend-text" style="color: ' + legendTextColor + ';">Electrode ' + (i + 1) + '</span>';
                
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
            items.forEach(function(item) {
                var idx = parseInt(item.dataset.electrodeIndex, 10);
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
            items.forEach(function(item) {
                var idx = parseInt(item.dataset.electrodeIndex, 10);
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
            chargesDiv.className = 'stl-overlay stl-charges';
            chargesDiv.style.cssText = 'position: absolute; top: 10px; left: 10px; background: ' + legendBg + '; padding: 12px; border-radius: 8px; font-size: 12px; max-width: 200px;';
            chargesDiv.innerHTML = '<strong style="color: ' + legendTextColor + '; display: block; margin-bottom: 8px; border-bottom: 1px solid ' + legendBorderColor + '; padding-bottom: 6px;">Electrode Charges</strong>';
            
            var charges = modelCharges[modelKey];
            for (var i = 0; i < charges.length && i < electrodeData.length; i++) {
                if (electrodeData[i] !== null) {
                    var colorHex = '#' + electrodeColors[i % electrodeColors.length].toString(16).padStart(6, '0');
                    var chargeItem = document.createElement('div');
                    chargeItem.style.cssText = 'padding: 4px 6px; margin: 2px 0; border-radius: 4px; display: flex; align-items: center; gap: 6px;';
                    chargeItem.innerHTML = '<span style="width: 10px; height: 10px; border-radius: 2px; background: ' + colorHex + '; flex: 0 0 auto;"></span><span style="color: ' + legendTextColor + ';">E' + (i + 1) + ': ' + charges[i] + '</span>';
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
        if (window.stlResizeObserver) {
            window.stlResizeObserver.disconnect();
            window.stlResizeObserver = null;
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
                if (trajLabel) trajLabel.textContent = trajCurrentTime.toFixed(3) + ' µs';
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
            if (trajLabel) trajLabel.textContent = trajCurrentTime.toFixed(3) + ' µs';
        }
    }
    
    function updatePositionTimeLabel() {
        var value = document.getElementById('posTimeValue');
        if (value) {
            if (posCurrentTime <= 0.001) {
                value.textContent = 'START';
            } else if (posCurrentTime >= posMaxTime - 0.001) {
                value.textContent = 'END';
            } else {
                value.textContent = posCurrentTime.toFixed(3) + ' µs';
            }
        }
    }
    
    // Initialize on page load
    document.addEventListener('DOMContentLoaded', initPositionTimeScrubber);
    
    // NOTE: The actual position-plot rendering (update2DTimePositions / updateVisibility /
    // getPositionAtTime) is implemented once, later in this file, driven by the uploaded/
    // preloaded model data (window.uploadedModelData + window.trajectoryData). See the
    // "POSITION PLOTS (Ion impact positions per timestep)" section below.
    
    // Theme toggle function
    function toggleTheme(persist) {
        var html = document.documentElement;
        var isLight = document.getElementById('themeToggle').checked;
        html.setAttribute('data-theme', isLight ? 'light' : 'dark');

        if (persist !== false) {
            try { localStorage.setItem('vmi-theme', isLight ? 'light' : 'dark'); } catch (e) {}
        }
        
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
        if (stlRenderer) {
            stlRenderer.setClearColor(isLight ? 0xf0f0f5 : 0x15151f, 1);
        }
        
        // Update all STL overlays, including the optional charge legend
        var stlContainer = document.getElementById('stlContainer');
        if (stlContainer) {
            var overlays = stlContainer.querySelectorAll('.stl-overlay');
            overlays.forEach(function(overlay) {
                overlay.style.background = isLight ? 'rgba(255,255,255,0.94)' : 'rgba(0,0,0,0.88)';
                var strongEl = overlay.querySelector('strong');
                if (strongEl) {
                    strongEl.style.color = isLight ? '#333' : 'white';
                    strongEl.style.borderBottomColor = isLight ? 'rgba(0,0,0,0.2)' : 'rgba(255,255,255,0.3)';
                }
                var legendTexts = overlay.querySelectorAll('.legend-text, div > span:last-child');
                legendTexts.forEach(function(textEl) {
                    textEl.style.color = isLight ? '#333' : 'white';
                });
                var btnContainer = overlay.querySelector('div[style*="border-top"]');
                if (btnContainer) {
                    btnContainer.style.borderTopColor = isLight ? 'rgba(0,0,0,0.2)' : 'rgba(255,255,255,0.3)';
                }
            });
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
    // Center of the trajectory data itself (per axis), used to align ion flight paths with
    // the chamber STL independently of the STL's own bounding box (simulation model was
    // smaller than the full chamber, so their natural axis origins don't coincide).
    var trajXOffset = 0;
    var trajYOffset = 0;
    var trajZOffset = 0;
    
    function loadTrajectoriesForModel(modelKey) {
        trajCurrentModel = modelKey;
        trajAnimating = false;
        trajCurrentTime = 0;
        trajMaxTime = 0;
        setTrajectoryPlayButton(false);
        
        // Clear existing trajectories
        trajIonMeshes.forEach(function(mesh) { if (stlScene) stlScene.remove(mesh); });
        trajTrailLines.forEach(function(line) { if (stlScene) stlScene.remove(line); });
        trajIonMeshes = [];
        trajTrailLines = [];
        
        if (!trajectoryData[modelKey]) {
            document.getElementById('trajIonCount').textContent = 'No trajectory data';
            document.getElementById('trajPlayBtn').disabled = true;
            document.getElementById('trajScrubber').disabled = true;
            document.getElementById('trajTimeLabel').textContent = 'Unavailable';
            return;
        }

        document.getElementById('trajPlayBtn').disabled = false;
        document.getElementById('trajScrubber').disabled = false;
        
        var data = trajectoryData[modelKey];
        var ionColor = new THREE.Color(modelColors[modelKey] || '#00d4ff');
        
        // Find max time and the per-axis center of the trajectory cloud (min/max midpoint)
        trajMaxTime = 0;
        var ionCount = 0;
        var trajXMin = Infinity, trajXMax = -Infinity;
        var trajYMin = Infinity, trajYMax = -Infinity;
        var trajZMin = Infinity, trajZMax = -Infinity;
        Object.keys(data).forEach(function(ionKey) {
            var steps = data[ionKey];
            ionCount++;
            if (steps.length > 0) {
                var lastTof = steps[steps.length - 1].tof;
                if (lastTof > trajMaxTime) trajMaxTime = lastTof;
                for (var s = 0; s < steps.length; s++) {
                    if (steps[s].x < trajXMin) trajXMin = steps[s].x;
                    if (steps[s].x > trajXMax) trajXMax = steps[s].x;
                    if (steps[s].y < trajYMin) trajYMin = steps[s].y;
                    if (steps[s].y > trajYMax) trajYMax = steps[s].y;
                    if (steps[s].z < trajZMin) trajZMin = steps[s].z;
                    if (steps[s].z > trajZMax) trajZMax = steps[s].z;
                }
            }
        });
        // Align to the chamber STL's bounding box (electrodes 1-6, 8, ignoring electrode7):
        // X and Z centered on the reference geometry, Y pinned to its rightmost (max) edge.
        var ref = window.trajAlignRef || { centerX: 0, centerZ: 0, maxY: 0 };
        var trajXCenter = (trajXMin <= trajXMax) ? (trajXMin + trajXMax) / 2 : 0;
        var trajZCenter = (trajZMin <= trajZMax) ? (trajZMin + trajZMax) / 2 : 0;
        trajXOffset = trajXCenter - ref.centerX;
        trajZOffset = trajZCenter - ref.centerZ;
        trajYOffset = (trajYMin <= trajYMax) ? (trajYMax - ref.maxY) : 0;
        
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
        document.getElementById('trajIonCount').textContent = ionCount + ' ions · ' + trajMaxTime.toFixed(2) + ' µs';
        document.getElementById('trajScrubber').max = trajMaxTime * 1000;
        document.getElementById('trajScrubber').value = 0;
        document.getElementById('trajTimeLabel').textContent = '0.000 µs';
         
        // Reset and show initial state
        trajCurrentTime = 0;
        trajAnimating = false;
        setTrajectoryPlayButton(false);
        updateTrajectoryPositions(0);
    }
    
    function toggleGridVisibility() {
        if (window.stlGridHelper) {
            window.stlGridHelper.visible = !window.stlGridHelper.visible;
        }
    }

    function updateClipping() {
        var ranges = window.clipRanges || { x: 250, y: 100, z: 100 };
        var xPct = parseFloat(document.getElementById('clipX').value);
        var yPct = parseFloat(document.getElementById('clipY').value);
        var zPct = parseFloat(document.getElementById('clipZ').value);
        
        // Convert percentage (-100..100) to actual mm based on the model's own bounding box
        var xVal = (xPct / 100) * ranges.x;
        var yVal = (yPct / 100) * ranges.y;
        var zVal = (zPct / 100) * ranges.z;
        
        // Update clipping plane positions
        if (clipPlaneX) clipPlaneX.constant = xVal;
        if (clipPlaneY) clipPlaneY.constant = yVal;
        if (clipPlaneZ) clipPlaneZ.constant = zVal;
        
        // Update labels
        document.getElementById('clipXLabel').textContent = xPct >= 100 ? 'Off' : xVal.toFixed(0) + 'mm';
        document.getElementById('clipYLabel').textContent = yPct >= 100 ? 'Off' : yVal.toFixed(0) + 'mm';
        document.getElementById('clipZLabel').textContent = zPct >= 100 ? 'Off' : zVal.toFixed(0) + 'mm';
    }
    
    function resetClipping() {
        document.getElementById('clipX').value = 0;
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
                    // All three axes are aligned using the trajectory's own center (trajXOffset/
                    // trajYOffset/trajZOffset) instead of the chamber STL's bounding box, since the
                    // simulation was run on a smaller model than the full chamber STL.
                    mesh.position.set(step.x - trajXOffset, step.y - trajYOffset, step.z - trajZOffset);
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
                    
                    for (var i = 0; i <= stepIdx; i++) {
                        // All axes aligned via trajXOffset/trajYOffset/trajZOffset (see above) so the
                        // trajectory cloud and the chamber STL share the same X=0/Y=0/Z=0 origin.
                        positions[i * 3] = steps[i].x - trajXOffset;
                        positions[i * 3 + 1] = steps[i].y - trajYOffset;
                        positions[i * 3 + 2] = steps[i].z - trajZOffset;
                        
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
        if (trajMaxTime <= 0) {
            showToast('No trajectory animation is available for this model.', 'error');
            return;
        }
        trajAnimating = !trajAnimating;
        setTrajectoryPlayButton(trajAnimating);
        
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
            setTrajectoryPlayButton(false);
        }
        
        updateTrajectoryPositions(trajCurrentTime);
        
        // Sync 2D position plot
        update2DTimePositions(trajCurrentTime);
        
        // Update 3D UI
        var scrubber = document.getElementById('trajScrubber');
        var label = document.getElementById('trajTimeLabel');
        if (scrubber) scrubber.value = trajCurrentTime * 1000;
        if (label) label.textContent = trajCurrentTime.toFixed(3) + ' µs';
        
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
            if (label) label.textContent = trajCurrentTime.toFixed(3) + ' µs';
            
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
        var scrubber = document.getElementById('trajScrubber');
        var label = document.getElementById('trajTimeLabel');
        setTrajectoryPlayButton(false);
        if (scrubber) scrubber.value = 0;
        if (label) label.textContent = '0.000 µs';
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
            if (selectedModels.length === 0) {
                if (modelKeys.length > 0) {
                    showToast('Select at least one model before exporting.', 'error');
                    return;
                }
                selectedModels = Object.keys(window.uploadedModelData || {});
            }

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
                    showToast('Detector plot exported.', 'success');
                }).catch(function(err){
                    showToast('Detector plot export failed: ' + err.message, 'error');
                    console.error(err);
                });
            } else {
                showToast('Export is available while the app is running in HTTP mode.', 'error');
            }
        }
        
        function downloadPositions(format) {
            // Legacy function for backward compatibility
            downloadDetectorPositions(format);
        }

        function download3D() {
            var canvas = document.querySelector('#stlContainer canvas');
            if (!canvas) { showToast('Open a 3D model before exporting.', 'error'); return; }
            // Compose a temporary canvas so we can add a legend overlay (charges)
            var w = canvas.width, h = canvas.height;
            var tmp = document.createElement('canvas'); tmp.width = w; tmp.height = h;
            var ctx = tmp.getContext('2d');
            // copy original
            ctx.drawImage(canvas, 0, 0);
            // draw charges legend if available
            var modelKey = window.currentSTLModelKey || window.trajCurrentModel || null;
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
            // Add a high-contrast annotation band that remains legible in both themes
            ctx.fillStyle = 'rgba(10,14,28,0.68)';
            ctx.fillRect(0, h - 46, w, 46);
            ctx.fillStyle = 'white'; ctx.font = '16px sans-serif';
            ctx.fillText('0 V region → increasingly negative potentials', 14, h - 17);
            var dataUrl = tmp.toDataURL('image/png');
            downloadDataUrl(dataUrl, 'vmi_3d.png');
            showToast('3D view exported.', 'success');
        }


        // ===== MODEL UPLOAD HANDLERS =====
        async function handleModelUpload() {
            const modelName = document.getElementById('modelNameInput').value.trim();
            const files = document.getElementById('modelFolderInput').files;
            
            if (files.length === 0) {
                setUploadStatus('error', 'Choose a model folder first.');
                return;
            }
            
            const formData = new FormData();
            if (modelName) formData.append('model_name', modelName);
            
            // Add all files from folder
            for (let file of files) {
                formData.append('files', file);
            }
            
            setUploadStatus('loading', 'Loading model data…');
            
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
                    setUploadStatus('error', 'The server returned an invalid response.');
                    return;
                }
                
                if (response.ok) {
                    setUploadStatus('success', 'Model loaded successfully.', data.warnings || []);
                    document.getElementById('modelNameInput').value = '';
                    document.getElementById('modelFolderInput').value = '';
                    document.getElementById('modelFolderLabel').textContent = 'No folder selected';
                    
                    // Refresh uploaded models list
                    setTimeout(loadUploadedModels, 500);
                } else {
                    const errorMessage = data.error || (data.errors || []).join(' ') || 'The model could not be loaded.';
                    setUploadStatus('error', errorMessage, data.warnings || []);
                }
            } catch (err) {
                console.error('Upload error:', err);
                setUploadStatus('error', 'Upload failed: ' + err.message);
            }
        }
        
        // Auto-fill model name from folder name
        document.getElementById('modelFolderInput').addEventListener('change', function(e) {
            if (this.files.length > 0) {
                // Get the folder path from first file
                const firstFile = this.files[0];
                const filePath = firstFile.webkitRelativePath || firstFile.name;
                const folderName = filePath.split('/')[0];
                const folderLabel = document.getElementById('modelFolderLabel');
                if (folderLabel) {
                    folderLabel.textContent = folderName + ' · ' + this.files.length + (this.files.length === 1 ? ' file' : ' files');
                    folderLabel.title = folderName;
                }
                
                if (folderName && !document.getElementById('modelNameInput').value.trim()) {
                    document.getElementById('modelNameInput').value = folderName;
                }
            } else {
                document.getElementById('modelFolderLabel').textContent = 'No folder selected';
            }
        });
        document.querySelector('.file-picker-button').addEventListener('keydown', function(event) {
            if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                document.getElementById('modelFolderInput').click();
            }
        });
        
        // ===== UPLOADED MODEL PLOT STATE =====
        window.uploadedModelData = {};   // key -> {y:[], z:[], color, name}
        window.stlDataCache = {};        // key -> [b64, b64, ...]  (per electrode)

        async function openSTLViewerUploaded(modelKey, modelName) {
            closeModelDrawer();
            // Fetch STL data from server if not cached
            if (!window.stlDataCache[modelKey]) {
                try {
                    const resp = await fetch('/get_stl_data/' + modelKey);
                    if (resp.ok) {
                        const d = await resp.json();
                        window.stlDataCache[modelKey] = d.stl_array || [];
                    } else {
                        throw new Error('HTTP ' + resp.status);
                    }
                } catch(e) {
                    showToast('Could not load the 3D data: ' + e.message, 'error');
                    return;
                }
            }
            // Temporarily inject into stlDataBase64 so openSTLViewer can find it
            stlDataBase64[modelKey] = window.stlDataCache[modelKey];
            openSTLViewer(modelKey, modelName);
        }

        // ============================================================================
        // POSITION PLOTS - ion positions per timestep / detector impact positions
        // ----------------------------------------------------------------------------
        // renderUploadedPlot(time) draws two aligned 2D views (top: YZ, bottom: ZX) of
        // where every checked model's ions are located at the given simulation time.
        //
        //   time === -1 (or the scrubber is at its rightmost/"End" position)
        //       -> the authoritative final impact positions from the result file are
        //          used, i.e. exactly where each ion actually landed on the detector.
        //   0 <= time < max flight time
        //       -> each ion's position is interpolated from its recorded trajectory
        //          steps (ions that already reached the detector earlier simply stay
        //          at their last known/impact position).
        // ============================================================================

        // Fixed axis range, computed once from the final impact-position data so the
        // plot doesn't rescale while scrubbing through time.
        window.plotAxisRange = null;

        // Finds an ion's position at (or just after) targetTime. If the ion's flight
        // already ended before targetTime, its last recorded position (its detector
        // impact position) is returned instead.
        function getPositionAtTime(ionSteps, targetTime) {
            if (!ionSteps || ionSteps.length === 0) return null;
            for (let i = 0; i < ionSteps.length; i++) {
                if (ionSteps[i].tof >= targetTime) return ionSteps[i];
            }
            return ionSteps[ionSteps.length - 1];
        }

        // Square, padded axis range covering every model's final detector impact positions.
        function computeAxisRange() {
            const modelData = window.uploadedModelData || {};
            let allX = [], allY = [], allZ = [];
            for (const m of Object.values(modelData)) {
                if (m.x) allX = allX.concat(Array.from(m.x));
                if (m.y) allY = allY.concat(Array.from(m.y));
                if (m.z) allZ = allZ.concat(Array.from(m.z));
            }
            if (allX.length === 0 && allY.length === 0 && allZ.length === 0) return null;

            const pad = 5;
            const rangeOf = (arr) => arr.length ? [Math.min(...arr) - pad, Math.max(...arr) + pad] : [0, 0];
            const [xMin, xMax] = rangeOf(allX);
            const [yMin, yMax] = rangeOf(allY);
            const [zMin, zMax] = rangeOf(allZ);

            // Use one common square size (largest span) so the YZ/ZX plots keep an equal aspect ratio.
            const size = Math.max(xMax - xMin, yMax - yMin, zMax - zMin, 1);
            const xCtr = (xMin + xMax) / 2, yCtr = (yMin + yMax) / 2, zCtr = (zMin + zMax) / 2;
            return {
                x: [xCtr - size / 2, xCtr + size / 2],
                y: [yCtr - size / 2, yCtr + size / 2],
                z: [zCtr - size / 2, zCtr + size / 2]
            };
        }

        // Returns {x:[], y:[], z:[]} for a model at a given timestep.
        // time < 0, undefined, or at/after the global max flight time -> real detector
        // impact positions (authoritative "end" data from the result file).
        // Otherwise -> per-ion trajectory position interpolated at that time.
        function getModelPositions(key, time) {
            const modelData = window.uploadedModelData || {};
            const trajData  = window.trajectoryData   || {};
            const m = modelData[key];
            if (!m) return { x: [], y: [], z: [] };

            const maxTime = (typeof posMaxTime !== 'undefined' && posMaxTime > 0) ? posMaxTime : Infinity;
            const atDetector = (time === undefined || time < 0 || time >= maxTime - 1e-6);

            const traj = trajData[key];
            if (traj && !atDetector) {
                const xs = [], ys = [], zs = [];
                Object.keys(traj).sort((a, b) => parseInt(a) - parseInt(b)).forEach(ionKey => {
                    const pos = getPositionAtTime(traj[ionKey], time);
                    if (pos) { xs.push(pos.x); ys.push(pos.y); zs.push(pos.z); }
                });
                return { x: xs, y: ys, z: zs };
            }
            // End / detector impact positions (authoritative, from the result file)
            return { x: Array.from(m.x || []), y: Array.from(m.y || []), z: Array.from(m.z || []) };
        }

        function renderUploadedPlot(time) {
            if (time === undefined) time = -1;
            const container = document.getElementById('plotly-container');
            if (!container) return;

            const modelData = window.uploadedModelData || {};
            const isLight   = document.documentElement.getAttribute('data-theme') === 'light';

            if (!window.plotAxisRange && Object.keys(modelData).length > 0) {
                window.plotAxisRange = computeAxisRange();
            }
            const axRange     = window.plotAxisRange;
            const gridColor   = isLight ? 'rgba(0,0,0,0.1)' : 'rgba(100,100,120,0.3)';
            const legendStyle = { bgcolor: isLight ? 'rgba(255,255,255,0.9)' : 'rgba(30,30,40,0.9)',
                                   font: { color: isLight ? '#333' : '#e0e0e0' } };

            const mainLayout = {
                // ZX plane: view of the ion impact positions on the detector
                xaxis: {
                    title: 'Z Position [mm]', gridcolor: gridColor,
                    range: axRange ? axRange.z : undefined,
                    domain: [0, 1], fixedrange: true, constrain: 'domain'
                },
                yaxis: {
                    title: 'X Position [mm]', gridcolor: gridColor,
                    range: axRange ? axRange.x : undefined,
                    domain: [0, 1], scaleanchor: 'x', scaleratio: 1, fixedrange: true
                },
                autosize: true,
                height: 480,
                uirevision: 'position-plot',
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: isLight ? 'rgba(250,250,255,0.9)' : 'rgba(25,25,35,0.8)',
                legend: legendStyle,
                margin: { l: 55, r: 20, t: 20, b: 55 }
            };

            if (Object.keys(modelData).length === 0) {
                Plotly.react(container, [{ x: [], y: [], mode: 'markers', type: 'scatter', name: '', marker: { size: 6 } }],
                    Object.assign({}, mainLayout, { annotations: [{ text: 'No model data loaded',
                        xref: 'paper', yref: 'paper', x: 0.5, y: 0.5, showarrow: false, font: { size: 16, color: '#666' } }] }));
                return;
            }

            const checkedKeys = Object.keys(modelData).filter(k => {
                const cb = document.getElementById('cb_' + k);
                return cb && cb.checked;
            });

            // --- ZX position traces ---
            const traces = checkedKeys.map(key => {
                const m   = modelData[key];
                const pos = getModelPositions(key, time);
                return { x: pos.z, y: pos.x, mode: 'markers', type: 'scatter', name: m.name,
                    marker: { size: 6, color: m.color, opacity: 0.8 }, xaxis: 'x', yaxis: 'y' };
            });

            Plotly.react(container,
                traces.length ? traces : [{ x: [], y: [], mode: 'markers', type: 'scatter', xaxis: 'x', yaxis: 'y' }],
                mainLayout, { responsive: true });
        }

        // Re-render the position plots at a given time (called by the 3D-viewer's
        // trajectory scrubber/animation to keep both views in sync).
        function update2DTimePositions(time) {
            renderUploadedPlot(time);
        }

        // Re-render the position plots at the currently selected time (called when a
        // model checkbox is toggled).
        function updateVisibility() {
            const t = (typeof posCurrentTime !== 'undefined') ? posCurrentTime : -1;
            renderUploadedPlot(t);
        }

        async function loadUploadedModels() {
            const container = document.getElementById('modelSelectionList');

            try {
                const response = await fetch('/get_models');
                if (!response.ok) throw new Error('HTTP ' + response.status);
                const data = await response.json();

                if (!data.uploaded_models || data.uploaded_models.length === 0) {
                    const emptyState = document.createElement('p');
                    emptyState.className = 'empty-state';
                    emptyState.textContent = 'No models loaded yet.';
                    container.replaceChildren(emptyState);
                    updateModelSelectionSummary();
                    renderUploadedPlot();  // Show empty axes
                    return;
                }

                // Preserve selections while rebuilding the list after an upload.
                const previousSelections = new Map(
                    Array.from(container.querySelectorAll('input[type="checkbox"]')).map(box => [box.id.slice(3), box.checked])
                );
                const fragment = document.createDocumentFragment();

                for (const model of data.uploaded_models) {
                    const card = document.createElement('article');
                    card.className = 'model-item';

                    const main = document.createElement('div');
                    main.className = 'model-item-main';

                    const label = document.createElement('label');
                    label.className = 'model-toggle';

                    const checkbox = document.createElement('input');
                    checkbox.type = 'checkbox';
                    checkbox.id = 'cb_' + model.key;
                    checkbox.checked = previousSelections.has(model.key) ? previousSelections.get(model.key) : true;
                    checkbox.addEventListener('change', function() {
                        updateModelSelectionSummary();
                        updateVisibility();
                    });

                    const swatch = document.createElement('span');
                    swatch.className = 'model-color';
                    swatch.style.backgroundColor = model.color;
                    swatch.setAttribute('aria-hidden', 'true');

                    const name = document.createElement('span');
                    name.className = 'model-name';
                    name.textContent = model.name;
                    name.title = model.name;

                    label.append(checkbox, swatch, name);
                    main.appendChild(label);

                    if (model.has_stls) {
                        const viewButton = document.createElement('button');
                        viewButton.type = 'button';
                        viewButton.className = 'btn-3d';
                        viewButton.innerHTML = '<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="m12 3 8 4.5v9L12 21l-8-4.5v-9L12 3Z M4.4 7.7 12 12l7.6-4.3M12 12v9"/></svg><span>3D view</span>';
                        viewButton.addEventListener('click', function() {
                            openSTLViewerUploaded(model.key, model.name);
                        });
                        main.appendChild(viewButton);
                    }

                    const meta = document.createElement('div');
                    meta.className = 'model-meta';
                    [
                        ['Results', true],
                        ['Animation', Boolean(model.has_trajectory)],
                        ['3D', Boolean(model.has_stls)]
                    ].forEach(function(status) {
                        const pill = document.createElement('span');
                        pill.className = 'status-pill' + (status[1] ? '' : ' is-missing');
                        pill.textContent = status[1] ? status[0] : 'No ' + status[0].toLowerCase();
                        meta.appendChild(pill);
                    });

                    card.append(main, meta);
                    fragment.appendChild(card);
                }

                container.replaceChildren(fragment);
                updateModelSelectionSummary();

                window.modelKeys = data.uploaded_models.map(m => m.key);
                window.trajectoryData = window.trajectoryData || {};
                window.modelColors = window.modelColors || {};

                // Load full model data (positions + trajectories)
                for (let model of data.uploaded_models) {
                    window.modelColors[model.key] = model.color;
                    try {
                        console.log('Loading data for model:', model.key);
                        const modelResp = await fetch('/get_model_data/' + model.key);
                        if (modelResp.ok) {
                            const modelData = await modelResp.json();
                            console.log('  Model data received:', modelData);
                            // Store end positions for plot
                            const endData = (modelData.data || {}).end || {};
                            console.log('  endData:', endData);
                            window.uploadedModelData[model.key] = {
                                x: endData.x || [],
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
                            console.error('  Request failed with status', modelResp.status);
                        }
                    } catch(e) {
                        console.warn('Could not load data for', model.key, e);
                    }
                }

                console.log('window.uploadedModelData final:', window.uploadedModelData);

                // Reset fixed axis range so it gets recomputed from new data
                window.plotAxisRange = null;

                // Recompute the max ion flight time from the freshly loaded trajectory data.
                // NOTE: this deliberately reuses the shared global `posMaxTime` (declared once,
                // near the time-scrubber controls) rather than a local shadow, so the scrubber's
                // "End" label and the detector-impact detection in getModelPositions() always
                // agree with the data that's actually loaded (preloaded demos or uploads alike).
                posMaxTime = 0;
                for (const key of Object.keys(window.trajectoryData || {})) {
                    for (const ionKey of Object.keys(window.trajectoryData[key])) {
                        const steps = window.trajectoryData[key][ionKey];
                        if (steps && steps.length > 0) {
                            const last = steps[steps.length-1].tof;
                            if (last > posMaxTime) posMaxTime = last;
                        }
                    }
                }
                const scrubber = document.getElementById('posTimeScrubber');
                if (scrubber && posMaxTime > 0) {
                    scrubber.max   = posMaxTime * 1000;
                    scrubber.value = posMaxTime * 1000;
                }
                posCurrentTime = posMaxTime > 0 ? posMaxTime : -1;
                updatePositionTimeLabel();

                // Show end positions initially (time = -1)
                renderUploadedPlot(-1);

                // Auto-open 3D viewer for first model that has STL data
                const firstWithStl = data.uploaded_models.find(m => m.has_stls);
                if (firstWithStl) {
                    openSTLViewerUploaded(firstWithStl.key, firstWithStl.name);
                }

            } catch (err) {
                const errorState = document.createElement('p');
                errorState.className = 'empty-state';
                errorState.textContent = 'Could not load models: ' + err.message;
                container.replaceChildren(errorState);
                updateModelSelectionSummary();
                showToast('Could not load the model list.', 'error');
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
    print('Open your browser and go to:')
    print('  http://127.0.0.1:5000/')
    print('  or')
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
            response['errors'].append('No files were uploaded.')
            return jsonify(response), 400
        
        files = request.files.getlist('files')
        if not files:
            response['errors'].append('No files were selected.')
            return jsonify(response), 400
        
        model_name = (request.form.get('model_name') or 'Uploaded Model').strip()
        model_slug = re.sub(r'[^A-Za-z0-9_-]+', '_', model_name).strip('_') or 'model'
        
        # Read uploaded files into memory
        model_files = {}
        for file in files:
            if file.filename:
                try:
                    content = file.read()
                    filename = Path(file.filename).name.lower()
                    model_files[filename] = content if filename.endswith('.stl') else content.decode('utf-8', errors='ignore')
                except Exception as e:
                    response['warnings'].append(f'File {file.filename} could not be read: {str(e)}')
        
        if not model_files:
            response['errors'].append('None of the selected files could be read.')
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
        
        model_key = f"uploaded_{len(UPLOADED_MODELS)}_{model_slug}"
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
    def _electrode_sort_key(name):
        m = re.search(r'(\d+)', name)
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
            # Match the interactive detector plot exactly: Z horizontal, X vertical.
            x_raw = end_data.get('z', [])
            y_raw = end_data.get('x', [])
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
            legend_space = 0.10 * span
            ax.set_xlim(x_min - pad, x_max +legend_space)
            ax.set_ylim(y_min - pad, y_max +legend_space)

        ax.set_xlabel('Z Position [mm]', fontsize=20, fontweight='bold')
        ax.set_ylabel('X Position [mm]', fontsize=20, fontweight='bold')
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
        ax.legend(ordered_h, ordered_l, frameon=True, fontsize=17, loc='lower center',bbox_to_anchor=(0.5, 1.02), ncol=2)
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

# Runs on import too (required for gunicorn, which never enters the __main__ block below)
preload_demo_models()
main()  # regenerates vmi_multi_comparison.html from the current source on every process start

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
preload_demo_models()
