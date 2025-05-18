# TexasSolver Integration with LLM-Poker-Solver

This document explains how to use TexasSolver as a baseline solver to verify LLM-generated poker strategies.

## Overview

TexasSolver is an efficient, open-source solver for Texas Hold'em poker. This integration provides:

1. A built console version of TexasSolver for running game tree solves
2. A Python bridge (`texas_solver.py`) that interfaces with the solver CLI
3. Utilities for comparing LLM-generated strategies with GTO (Game Theory Optimal) solutions

## Setup

### Prerequisites

- C++ compiler with C++17 support
- CMake 3.15 or newer
- Python 3.6 or newer
- pip for installing Python dependencies

### Building TexasSolver

The solver has been modified to work without OpenMP on macOS systems. To build it:

```bash
cd external/TexasSolver
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
make -j$(sysctl -n hw.ncpu)
```

This will produce the binary `external/TexasSolver/build/console_solver`.

### Installing Python Dependencies

```bash
pip install numpy pandas matplotlib
```

## Using the Python Bridge

The `texas_solver.py` file provides a Python interface to the TexasSolver CLI. Here's a simple example:

```python
from texas_solver import TexasSolverBridge, SolverConfig

# Create a solver instance
solver = TexasSolverBridge()

# Configure the solver
config = SolverConfig(
    iterations=1000,
    board="Ah7d2c",
    range_oop="AK,AQ,AJ,AT,A9,A8,A7,A6,A5,A4,A3,A2,KK,QQ,JJ,TT,99,88,77,66,55,44,33,22",
    range_ip="AK,AQ,AJ,AT,A9,A8,A7,A6,A5,A4,A3,A2,KK,QQ,JJ,TT,99,88,77,66,55,44,33,22",
)
solver.set_config(config)

# Run the solver
result = solver.run_solver()

# Get results
ev = solver.get_ev()
oop_strategy = solver.get_strategy("oop", "flop")
ip_strategy = solver.get_strategy("ip", "flop")
```

## Configuration Options

The `SolverConfig` class supports the following parameters:

- `iterations`: Maximum number of iterations for the solver (default: 1000)
- `accuracy`: Target accuracy for early stopping (default: 0.001)
- `board`: Board cards in string format (e.g., "Ah7d2c")
- `pot_sizes`: Starting pot sizes for both players (default: [100, 100])
- `range_oop`: Hand range for out-of-position player
- `range_ip`: Hand range for in-position player
- `bet_sizes`: Betting sizes as % of pot for each street
- `raise_sizes`: Raise sizes as multipliers for each street
- `donk_sizing`: Donk bet sizes as % of pot

## Comparing with LLM Strategies

To compare LLM-generated strategies with GTO solutions:

1. Run the TexasSolver on a specific game configuration
2. Generate a strategy for the same situation using the LLM
3. Use the utility functions in the Python bridge to analyze the differences:
   - Exploitability analysis
   - Strategy visualization
   - EV comparison

## Limitations

- The current integration supports standard Texas Hold'em scenarios but may not handle all specialized bet sizing or game configurations
- Solving time increases significantly with complex game trees
- Memory usage can be substantial for deep game trees or wide hand ranges

## Troubleshooting

- **Compilation errors**: Make sure you have the correct dependencies installed
- **Runtime errors**: Check the solver's output for specific error messages
- **Performance issues**: For large game trees, increase available memory or reduce the complexity of the configuration

## References

- [TexasSolver GitHub Repository](https://github.com/bupticybee/TexasSolver)
- [Console version documentation](https://github.com/bupticybee/TexasSolver/tree/console) 