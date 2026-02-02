# Chat-Rten
## A RAG Chatbot starting from robertknight/rten

### To set up:
- find a model (currently testing with qwen2-O.5b and 1.5b)
- convert it to onnx, something like
```
optimum-cli export onnx --model Qwen/Qwen3-1.7B qwen3-1.7b/
```
- put the resulting directory inside `models` (which is .gitignore'd so back it up somewhere else)

To quantize an existing model:
```
python3 ort-quantize.py nbits ../models/qwen3-1.7b/model.onnx # tools directory
```

### To run:
```
cd rag_chat
cargo run --release --bin rag_chat ../models/qwen2-0.5b/model.onnx ../models/qwen2-0.5b/tokenizer.json
# or
# cargo run --release --bin rag_chat ../models/qwen2.5-1.5b/model.onnx ../models/qwen2.5-1.5b/tokenizer.json
# or
# cargo run --release --bin rag_chat ../models/qwen3-1.7b/model.onnx ../models/qwen3-1.7b/tokenizer.json
```

The upstream README is [here](./README_UPSTREAM.md).
