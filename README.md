# OpenPhonemizer to ONNX Converter

This script automates the download and conversion of the **OpenPhonemizer** model into the **ONNX** format. 

The resulting files are specifically optimized for use with [piper-unity (no-espeak)](https://github.com/lookbe/piper-no-espeak-unity), allowing for a fully permissive, GPL-free Text-to-Speech stack in Unity.

## 🎯 Purpose
OpenPhonemizer is a deep-learning-based G2P (Grapheme-to-Phoneme) engine. Since Unity environments (especially mobile and consoles) benefit from specialized inference engines, this script:
1. Downloads the latest PyTorch weights from the OpenPhonemizer source.
2. Traces the model and exports it to `.onnx`.
3. Handles the export of the character tokenizer and phoneme dictionary.
