from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_hf_token

def test_llama_load():
    token = get_hf_token()
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, token=token, load_in_8bit=True, device_map="auto"
    )
    out = model.generate(**tokenizer("hello", return_tensors="pt").to("mps"), max_new_tokens=1)
    assert out.shape[1] == 2  # input + 1 generated token 