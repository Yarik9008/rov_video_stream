#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import sys
import os

def check_pyinstaller():
    try:
        import PyInstaller
        print("PyInstaller found")
        return True
    except ImportError:
        print("PyInstaller not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        return True

def build_executable(script_name, exe_name):
    print(f"\nBuilding {exe_name} from {script_name}...")
    cmd = [
        "pyinstaller",
        "--onefile",
        "--name", exe_name,
        "--clean",
        script_name
    ]
    try:
        subprocess.check_call(cmd)
        print(f"OK: {exe_name} built successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error building {exe_name}: {e}")
        return False

def main():
    print("=" * 50)
    print("Building executables")
    print("=" * 50)
    
    if not check_pyinstaller():
        print("Failed to install PyInstaller")
        return 1
    
    scripts = [
        ("server.py", "server"),
        ("client.py", "client"),
        ("client_qt.py", "client_qt"),
    ]
    
    success_count = 0
    for script, exe_name in scripts:
        if not os.path.exists(script):
            print(f"Warning: {script} not found, skipping...")
            continue
        
        if build_executable(script, exe_name):
            success_count += 1
    
    print("\n" + "=" * 50)
    print(f"Build complete: {success_count}/{len(scripts)} files")
    print("Executables are in the 'dist' folder")
    print("=" * 50)
    
    return 0 if success_count == len(scripts) else 1

if __name__ == "__main__":
    sys.exit(main())
