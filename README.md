# Chat-Rten
## A RAG Chatbot starting from robertknight/rten

### To set up:
- find a model (currently testing with qwen2-O.5b and 1.5b)
- convert it to onnx
- put the resulting directory inside `models` (which is .gitignore'd so back it up somewhere else)

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
