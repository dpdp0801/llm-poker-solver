# Script Documentation

This directory contains several scripts for analyzing and solving poker situations:

## Advanced Solver Demo

```bash
python scripts/advanced_solver_demo.py
```

This is the main interactive tool for comprehensive poker analysis. It provides:

- Preflop range lookup based on positions and actions
- Board texture analysis
- Dynamic bet sizing recommendations
- Integration with TexasSolver for solving
- Game tree navigation
- Hand-specific strategy analysis with realistic mixed strategies

## Simple Preflop Range Demo

```bash
python display_preflop_methods.py
```

A simpler tool that demonstrates the preflop range lookup functionality. It shows:

- Recommended actions for specific hands in various preflop scenarios
- Direct access to the preflop chart data

## Original Solver Demo

```bash
python scripts/solver_demo.py
```

The original TexasSolver integration demo. Less sophisticated than the advanced demo, but provides:

- Basic solver commands generation
- Simple strategy summarization

## Other Scripts

- `evaluate_ev.py`: For evaluating expected value of specific hands/ranges (placeholder)
- `run_solver_test.py`: Run tests on the solver (placeholder)
- `train_stageA.py`: For LLM training (placeholder)
- `train_stageB.py`: For LLM training (placeholder)

## Development

When extending these scripts, consider these interfaces:

- `PreflopLookup` from `preflop.py` for range lookup
- `TexasSolverBridge` from `texas_solver.py` for solver integration
- Utility functions from `utils.py` for hand/board analysis 