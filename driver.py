import llm_inference_engine as engine
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

prompt = "The capital of France is"
encoded = tokenizer(prompt)['input_ids']
e = engine.Engine(encoded)
output = e.forward_pass()

# Printing 
print(tokenizer.decode(output))