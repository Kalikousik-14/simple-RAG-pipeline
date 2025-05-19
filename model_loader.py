from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def load_local_llm():
    model_id = "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ" 
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", trust_remote_code=True)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)