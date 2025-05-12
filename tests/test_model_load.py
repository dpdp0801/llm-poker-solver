from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def test_llama_load():
    token = os.environ.get("HF_TOKEN")
    assert token, "set HF_TOKEN env var"
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, token=token, load_in_8bit=True, device_map="auto"
    )
    out = model.generate(**tokenizer("hello", return_tensors="pt").to("mps"), max_new_tokens=1)
    assert out.shape[1] == 2  # input + 1 generated token 