import torch
from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.utils import get_tokenizer


model, vocab = get_pytorch_kogpt2_model()
print(model)
print(vocab)
print("DONE!")