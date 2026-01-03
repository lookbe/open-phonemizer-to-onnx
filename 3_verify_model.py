import onnxruntime as ort
import torch
from openphonemizer import OpenPhonemizer
from dp.model.model import load_checkpoint
import argparse
import json
import re

def run_onnx_inference(onnx_path, preprocessor, text, lang='en_us'):
    # Replicate OpenPhonemizer's splitting logic
    punctuation = '().,:?!/–'
    punc_set = set(punctuation + '- ')
    punc_pattern = re.compile(f'([{punctuation + " "}])')
    
    split_text = [s for s in re.split(punc_pattern, text) if s]
    
    session = ort.InferenceSession(onnx_path)
    final_phonemes = []
    
    for word in split_text:
        if word in punc_set:
            final_phonemes.append(word)
            continue
            
        input_tokens = preprocessor.text_tokenizer(word, lang)
        input_tensor = torch.tensor(input_tokens).unsqueeze(0).long()
        
        inputs = {session.get_inputs()[0].name: input_tensor.numpy()}
        logits = session.run(None, inputs)[0]
        
        output_tokens = torch.tensor(logits).argmax(dim=-1)[0].tolist()
        
        # Simple dedup
        dedup = []
        for i, t in enumerate(output_tokens):
            if i == 0 or t != output_tokens[i-1]:
                dedup.append(t)
                
        phonemes = preprocessor.phoneme_tokenizer.decode(dedup, remove_special_tokens=True)
        final_phonemes.append(''.join(phonemes))
        
    return ''.join(final_phonemes)

def verify(model_pt, model_onnx, text):
    print(f"--- Verifying with text: '{text}' ---")
    
    # 1. OpenPhonemizer (PyTorch/Reference)
    print("Running OpenPhonemizer (PyTorch reference)...")
    ph = OpenPhonemizer(model_checkpoint=model_pt)
    res_pt = ph(text)
    print(f"PyTorch Result: {res_pt}")
    
    # 2. ONNX Verification
    print("Running ONNX Inference...")
    from dp.preprocessing.text import Preprocessor
    checkpoint = torch.load(model_pt, map_location='cpu', weights_only=False)
    preprocessor = checkpoint['preprocessor']
    res_onnx = run_onnx_inference(model_onnx, preprocessor, text)
    print(f"ONNX Result:    {res_onnx}")
    
    if res_pt == res_onnx:
        print("\n✅ SUCCESS: Results match!")
    else:
        print("\n❌ MISMATCH: Results differ.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify model versions")
    parser.add_argument('--pt', type=str, default='best_model.pt')
    parser.add_argument('--onnx', type=str, default='model.onnx')
    parser.add_argument('--text', type=str, default='Deep Learning is amazing')
    args = parser.parse_args()
    
    verify(args.pt, args.onnx, args.text)
