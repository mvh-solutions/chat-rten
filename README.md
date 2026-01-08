# Chat-Rten
## A RAG Chatbot starting from robertknight/rten

### To set up:
- find a model (currently testing with qwen2-O.5b)
- convert it to onnx
- put the resulting directory inside `models` (which is .gitignore'd so back it up somewhere else)

### To run:
```
cd rten-examples
cargo run --release --bin qwen2_chat ../models/qwen2-0.5b/model.onnx ../models/qwen2-0.5b/tokenizer.json
```

The upstream README is [here](./README_UPSTREAM.md).
