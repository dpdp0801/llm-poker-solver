#!/usr/bin/env python3
"""
Run the TexasSolver directly from the command line with a simple example.
This script is useful for verifying that the solver works independently of the Python wrapper.
"""

import os
import sys
import subprocess
import tempfile

def main():
    """Run the solver directly with a simple example."""
    # Path to the solver binary
    solver_path = os.path.join('external', 'TexasSolver', 'build', 'console_solver')
    
    # Check if the solver exists
    if not os.path.exists(solver_path):
        print(f"Error: Solver binary not found at {solver_path}")
        print("Please build the solver first:")
        print("cd external/TexasSolver && mkdir -p build && cd build && cmake .. && make")
        return
    
    # Create a simple input file
    commands = [
        "set_pot 100",
        "set_effective_stack 200",
        "set_board Ah,Kd,5c",
        "set_range_ip AA,KK,QQ,JJ,TT,99,88,77,66,55,AKs,AQs,AJs,ATs,A9s,A8s,A7s,A6s,A5s,A4s,A3s,KQs,KJs,KTs,QJs,JTs,AKo,AQo,AJo,ATo,KQo",
        "set_range_oop AA,KK,QQ,JJ,TT,99,88,77,66,55,44,33,22,AKs,AQs,AJs,ATs,A9s,A8s,A7s,A6s,A5s,A4s,A3s,A2s,KQs,KJs,KTs,K9s,K8s,K7s,K6s,K5s,K4s,QJs,QTs,JTs,T9s,98s,87s,76s,65s,54s,AKo,AQo,AJo,ATo,KQo,KJo,QJo",
        "set_bet_sizes ip,flop,bet,25,33,50,66,75,100,150",
        "set_bet_sizes oop,flop,bet,25,33,50,66,75,100,150",
        "build_tree",
        "set_accuracy 0.001",
        "set_max_iterations 1000",
        "start_solve",
        "dump_result solver_direct_results.json"
    ]
    
    # Write commands to a temporary file
    fd, input_file = tempfile.mkstemp(suffix='.txt')
    with os.fdopen(fd, 'w') as f:
        f.write('\n'.join(commands))
    
    print(f"Created input file: {input_file}")
    print("\n===== Solver Commands =====")
    for cmd in commands:
        print(f"> {cmd}")
    
    print("\n===== Running Solver Directly =====")
    print(f"Running: {solver_path} -i {input_file}")
    
    # Run the solver with the input file
    try:
        # This will display output in real-time
        process = subprocess.Popen(
            [solver_path, '-i', input_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        
        print("\n===== Solver Execution Complete =====")
        if process.returncode == 0:
            print("Solver completed successfully!")
            print(f"Results saved to solver_direct_results.json")
        else:
            print(f"Solver failed with return code {process.returncode}")
    
    except Exception as e:
        print(f"Error running solver: {e}")
    
    # Clean up the temporary file
    try:
        os.unlink(input_file)
    except:
        pass

if __name__ == '__main__':
    main() 