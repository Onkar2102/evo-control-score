#!/usr/bin/env python3
"""
Wrapper script to run M3 optimizer from the correct directory
"""

import os
import sys
import subprocess

def main():
    # Ensure we're in the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    
    # Build the command
    cmd = [sys.executable, "src/utils/m3_optimizer.py"] + sys.argv[1:]
    
    print(f"Running M3 optimizer from: {project_root}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)
    
    try:
        # Run the optimizer
        result = subprocess.run(cmd, check=True)
        print("-" * 60)
        print("M3 optimizer completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error running M3 optimizer: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nM3 optimizer interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main() 