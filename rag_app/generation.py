# Response generation
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .config import logger

def generate_response(
    query: str, 
    context: List[str], 
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    rag_cache: dict,
    device: torch.device
) -> str:
    if not query.strip() or not context:
        return "I need both a question and context to provide an answer."
    
    if model is None or tokenizer is None:
        return "The language model is not available. I can show you relevant passages but cannot generate a comprehensive answer."
    
    try:
        cache_key = f"response_{hash(query)}_{hash(tuple([c[:100] for c in context]))}"
        if cache_key in rag_cache:
            return rag_cache[cache_key]
            
        formatted_context = ""
        for i, chunk in enumerate(context):
            chunk_preview = chunk[:min(len(chunk), 1500)]  
            formatted_context += f"\nDocument excerpt {i+1}:\n{chunk_preview.strip()}\n"
            
        prompt = f"""<s>[INST] You have given context based on which provide answer to below question. Provide answer in short and to the point. Try to tell answer in 2 lines.
        
        Relevant document excerpts:
        {formatted_context}
        
        Question: {query}
        
        Instructions:
        1. Answer ONLY based on the provided document.
        2. Be concise but thorough, focusing on the most relevant information.
        3. If the answer isn't in the excerpts, explain that you don't have enough information.
        4. Do not make up information or draw from external knowledge.
        5. Include specific details from the document where relevant.
        [/INST]"""
        
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=350,
                temperature=0.5,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                early_stopping=True
            )
            
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        final_response = full_response.split("[/INST]")[-1].strip()
        
        if "</think>" in final_response:
            parts = final_response.split("</think>")
            final_response = parts[-1].strip()
        
        if final_response.lower().startswith("answer:"):
            final_response = final_response[7:].strip()
            
        rag_cache[cache_key] = final_response
        return final_response
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "I encountered an error while generating a response. This might be due to memory constraints. You can still view the relevant passages below."