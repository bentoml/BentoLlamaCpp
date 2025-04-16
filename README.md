## Installation

```bash
uv venv -p 3.11

# For M1 Mac
CMAKE_ARGS="-DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_APPLE_SILICON_PROCESSOR=arm64 -DGGML_METAL=on" pip install --upgrade --verbose --force-reinstall --no-cache-dir llama-cpp-python
# For Mac
CMAKE_ARGS="-DGGML_METAL=on" pip install --upgrade --verbose --force-reinstall --no-cache-dir llama-cpp-python

uv pip install -r pyproject.toml

bentoml serve
```

If you want to use different models:

```bash
bentoml serve -f qwq.yaml
```

It will use [Gemma 3](https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF) by default here.

## Deploy

```bash
bentoml deploy
```
