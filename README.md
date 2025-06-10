## Quick Start

### Prerequisites

- **Anaconda or Miniconda** 
- **Python 3.10-3.11**
- **24GB+ RAM**
- **10GB+ disk space** (for models)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/dpdp0801/llm-poker-solver.git
   cd llm-poker-solver
   ```

2. **Set up the environment**:
   ```bash
   conda env create -f environment.yaml
   conda activate poker-llm
   ```

3. **Download model weights**

paths for model weights:
out/stageA/stageA_bf16_r256_seq2048
out/stageB/systemB_bf16_r256
out/stageB/systemB_bf16_r256
out/stageB/systemB_bf16_r256

4. **Set up hugging face token in .env**
HF_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXX


#### GUI Application
```bash
# Launch the interactive GUI
./run_poker_solver.sh


## 🎮 How to Use the GUI

1. **Launch the application**: Run `./run_poker_solver.sh`
2. **Enter game state**: 
   - Set player positions (Hero/Villain)
   - Input hole cards (e.g., "As Kd")
   - Enter flop cards (e.g., "Jh Tc 9s")
3. **Analyze**: Click "Generate Prompt" and then "Analyze Hand" to get recommendations
4. **Results**: 
   - **Action recommendation** with bet sizing
   - **Hand categorization** (e.g., "overpair", "set with flush draw")
   - **Equity analysis** and range advantages
   - **Strategic reasoning** in plain English