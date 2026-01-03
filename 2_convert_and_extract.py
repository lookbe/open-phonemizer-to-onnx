import torch
import torch.nn as nn
from dp.model.model import create_model, ModelType
import argparse
import json
import os

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, text):
        return self.model.forward({'text': text})

def convert_and_extract(model_pt, model_onnx, dict_json, tokenizer_json):
    # Ensure Preprocessor class is visible for torch.load
    from dp.preprocessing.text import Preprocessor
    
    print(f"Loading checkpoint from {model_pt}...")
    checkpoint = torch.load(model_pt, map_location='cpu', weights_only=False)
    
    # 1. Extract Dictionary
    phoneme_dict = checkpoint.get('phoneme_dict', {})
    if phoneme_dict:
        print(f"Found phoneme dictionary with {len(phoneme_dict)} entries. Saving to {dict_json}...")
        with open(dict_json, 'w', encoding='utf-8') as f:
            json.dump(phoneme_dict, f, ensure_ascii=False, indent=2)
    
    # 2. Extract Tokenizer
    print(f"Extracting tokenizer config to {tokenizer_json}...")
    preprocessor = checkpoint['preprocessor']
    tokenizer_config = {
        'text_symbols': preprocessor.text_tokenizer.token_to_idx,
        'phoneme_symbols': preprocessor.phoneme_tokenizer.idx_to_token,
        'char_repeats': preprocessor.text_tokenizer.char_repeats,
        'languages': preprocessor.text_tokenizer.languages
    }
    with open(tokenizer_json, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)

    # 3. Convert to ONNX
    print("Exporting model to ONNX (single file)...")
    config = checkpoint['config']
    model_type = ModelType(config['model']['type'])
    model = create_model(model_type, config=config)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    wrapped_model = ModelWrapper(model)
    dummy_input = torch.randint(low=0, high=10, size=(1, 5), dtype=torch.long)
    
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        model_onnx,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['text'],
        output_names=['logits'],
        dynamic_axes={
            'text': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size', 1: 'sequence_length'}
        },
        external_data=False
    )
    print(f"ONNX model saved to {model_onnx}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert and Extract model data")
    parser.add_argument('--in_pt', type=str, default='best_model.pt')
    parser.add_argument('--out_onnx', type=str, default='model.onnx')
    parser.add_argument('--out_dict', type=str, default='phoneme_dict.json')
    parser.add_argument('--out_tok', type=str, default='tokenizer.json')
    args = parser.parse_args()
    
    convert_and_extract(args.in_pt, args.out_onnx, args.out_dict, args.out_tok)
