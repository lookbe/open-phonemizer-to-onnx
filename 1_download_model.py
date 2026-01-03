import os
import torch
import argparse
from huggingface_hub import hf_hub_download

def download_and_clean_model(repo_id, filename, output_file):
    """Downloads model from HF and removes training states for smaller size."""
    print(f"Downloading {filename} from {repo_id}...")
    try:
        # Download to a temporary path or use the requested filename
        downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=".")
        print(f"Successfully downloaded to: {downloaded_path}")
        
        print("Cleaning checkpoint (removing optimizer states)...")
        checkpoint = torch.load(downloaded_path, map_location='cpu', weights_only=False)
        
        # Keep only what's needed for inference
        clean_checkpoint = {
            'model': checkpoint['model'],
            'config': checkpoint['config'],
            'preprocessor': checkpoint['preprocessor'],
            'phoneme_dict': checkpoint.get('phoneme_dict', {}),
            'step': checkpoint.get('step', 0)
        }
        
        torch.save(clean_checkpoint, output_file)
        
        old_size = os.path.getsize(downloaded_path) / (1024 * 1024)
        new_size = os.path.getsize(output_file) / (1024 * 1024)
        print(f"Cleaned model saved to: {output_file}")
        print(f"Size reduced from {old_size:.2f} MB to {new_size:.2f} MB")
        
        # If the output file is different from downloaded, maybe remove the original?
        # For simplicity, we'll leave it or let the user decide.
        
    except Exception as e:
        print(f"Error during download or cleaning: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and clean DeepPhonemizer model")
    parser.add_argument('--repo', type=str, default='openphonemizer/ckpt', help='HF Repo ID')
    parser.add_argument('--file', type=str, default='best_model.pt', help='Model filename')
    parser.add_argument('--out', type=str, default='best_model.pt', help='Output cleaned path')
    args = parser.parse_args()
    
    download_and_clean_model(args.repo, args.file, args.out)
