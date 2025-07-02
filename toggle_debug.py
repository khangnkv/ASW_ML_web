#!/usr/bin/env python3
"""
Script to toggle debug prints on/off in the ML backend
Usage: python toggle_debug.py [on|off]
"""

import sys
import os
from pathlib import Path

def toggle_debug(enable=False):
    """Toggle debug prints in all relevant files"""
    
    # Files to modify
    files_to_modify = [
        'backend/app_workflow.py',
        'backend/model/predictor.py', 
        'preprocessing.py'
    ]
    
    target_value = "True" if enable else "False"
    
    for file_path in files_to_modify:
        if os.path.exists(file_path):
            print(f"Updating {file_path}...")
            
            # Read the file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Replace DEBUG_PRINTS value
            if 'DEBUG_PRINTS = True' in content:
                new_content = content.replace('DEBUG_PRINTS = True', f'DEBUG_PRINTS = {target_value}')
            elif 'DEBUG_PRINTS = False' in content:
                new_content = content.replace('DEBUG_PRINTS = False', f'DEBUG_PRINTS = {target_value}')
            else:
                print(f"Warning: Could not find DEBUG_PRINTS in {file_path}")
                continue
            
            # Write back the file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print(f"✓ Updated DEBUG_PRINTS = {target_value} in {file_path}")
        else:
            print(f"Warning: File {file_path} not found")
    
    status = "enabled" if enable else "disabled"
    print(f"\nDebug prints {status}!")
    print("Remember to restart your backend service for changes to take effect.")

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ['on', 'off']:
        print("Usage: python toggle_debug.py [on|off]")
        sys.exit(1)
    
    enable_debug = sys.argv[1] == 'on'
    toggle_debug(enable_debug)
