import mxnet as mx
from kogpt2.mxnet_kogpt2 import get_mxnet_kogpt2_model
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.utils import get_tokenizer

if mx.context.num_gpus() > 0:
  ctx = mx.gpu()
else:
  ctx = mx.cpu()
  
tok_path = get_tokenizer()
model, vocab = get_mxnet_kogpt2_model(ctx=ctx, cachedir="./cache")
tok = SentencepieceTokenizer(tok_path)
sent = '2019년 한해를 보내며,'
toked = tok(sent)
while True:
  input_ids = mx.nd.array([vocab[vocab.bos_token]]  + vocab[toked]).expand_dims(axis=0)
  pred = model(input_ids.as_in_context(ctx))[0]
  gen = vocab.to_tokens(mx.nd.argmax(pred, axis=-1).squeeze().astype('int').asnumpy().tolist())[-1]
  if gen == '</s>':
    break
  sent += gen.replace('▁', ' ')
  toked = tok(sent)
print(sent)