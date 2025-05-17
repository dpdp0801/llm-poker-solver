# LLM-GTO

GPT4 created solution for computing GTO (Game Theory Optimal) solutions for poker games.

## Setup

1. Install dependencies: 
```bash
./setup.sh
```

2. Set up Hugging Face Access Token:
   - Create a Hugging Face account if you don't have one
   - Generate an access token at https://huggingface.co/settings/tokens
   - Copy `.env.template` to `.env` and add your token:
     ```
     HF_TOKEN=your_huggingface_token_here
     ```
   - The code will automatically load this token using python-dotenv

3. Run tests:
```bash
python run_solver_test.py  # full end-to-end test
``` 