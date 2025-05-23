# LLM Poker Solver

A toolkit for poker analysis that combines preflop ranges with postflop solving capabilities, designed to help players improve their strategic decision-making in Texas Hold'em.

## Features

- **Preflop Range Analysis**: Access GTO-based preflop ranges for different positions and scenarios
- **Postflop Solving**: Integration with TexasSolver for running postflop simulations
- **Board Texture Analysis**: Automatic classification of board textures to inform betting strategies
- **Dynamic Bet Sizing**: Intelligent bet size recommendations based on board texture
- **Game Tree Navigation**: Explore decision points throughout a hand
- **Hand-Specific Analysis**: Get optimal play frequencies for specific hands in your range

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llm-poker-solver.git
cd llm-poker-solver
```

2. Build the TexasSolver binary:
```bash
cd external/TexasSolver
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
cd ../../..
```

## Usage

### Preflop Range Lookup

The `display_preflop_methods.py` script provides an interface to the preflop range lookup:

```bash
python display_preflop_methods.py
```

This will display recommended actions for various preflop scenarios.

### Advanced Solver Demo

For a more comprehensive analysis that includes board texture and postflop strategy, use the advanced solver demo:

```bash
python scripts/advanced_solver_demo.py
```

This interactive tool will:
1. Get preflop ranges based on your specified action sequence
2. Analyze board texture and recommend appropriate bet sizes
3. Run the solver to determine optimal strategies
4. Allow you to explore hand-specific strategies

## Project Structure

- `src/llm_poker_solver/preflop.py`: Preflop range lookup module
- `src/llm_poker_solver/texas_solver.py`: Python bridge to the TexasSolver
- `src/llm_poker_solver/utils.py`: Utility functions for poker analysis
- `scripts/advanced_solver_demo.py`: Interactive demo for comprehensive analysis
- `display_preflop_methods.py`: Simple preflop range lookup demo
- `external/TexasSolver/`: Third-party solver for postflop analysis

## Recent Improvements

1. **Fixed Hand Format Issues**: Resolved compatibility issues between range notation and specific card notation
2. **Improved Board Texture Analysis**: Added sophisticated board texture classification
3. **Dynamic Bet Sizing**: Intelligent bet size recommendations based on board characteristics
4. **Better Game Tree Navigation**: More realistic strategy generation with conditional decision-making
5. **Enhanced User Interface**: Interactive command-line interface with formatted card displays

## Contributing

Contributions are welcome! Some areas that could use improvement:

- More sophisticated board texture analysis
- Turn and river strategy implementation
- Web-based user interface
- Integration with real-time hand tracking software

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [TexasSolver](https://github.com/bupticybee/TexasSolver) for the underlying solving engine
- Various GTO resources for preflop ranges

