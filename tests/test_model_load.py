from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_poker_solver.utils import get_hf_token


def test_llama_load():
    token = get_hf_token()
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, token=token, load_in_8bit=True, device_map="auto"
    )
    out = model.generate(
        **tokenizer("hello", return_tensors="pt").to("cpu"), max_new_tokens=1
    )
    assert out.shape[1] == 2
