from __future__ import annotations

import os, bentoml, pydantic, json, fastapi, traceback, typing as t, annotated_types as ae

from starlette.responses import JSONResponse, StreamingResponse

with bentoml.importing():
  from llama_cpp import Llama, ChatCompletionFunction


class Message(pydantic.BaseModel):
  role: t.Literal['system', 'user', 'assistant']
  content: str


class BentoArgs(pydantic.BaseModel):
  # engine args
  model_id: str = 'ggml-org/gemma-3-4b-it-GGUF'
  filename: str = '*Q8_0.gguf'
  kwargs: dict[str, t.Any] = pydantic.Field(default_factory=dict)

  # inference args
  max_tokens: int = pydantic.Field(default_factory=lambda: int(os.environ.get('MAX_TOKENS', 2048)))
  temperature: float = pydantic.Field(default=0.6)

  # service args
  name: str = 'gemma3'
  cpu: int = 2
  memory: str = '16Gi'


bento_args = bentoml.use_arguments(BentoArgs)
openai_api_app = fastapi.FastAPI()


@openai_api_app.get('/models')
async def show_available_models():
  return {'data': [{'id': bento_args.model_id, 'object': 'model', 'created': 1686935002, 'owned_by': 'bentoml'}]}


@bentoml.asgi_app(openai_api_app, path='/v1')
@bentoml.service(
  name=f'bentollamacpp-{bento_args.name}-instruct-service',
  resources={'cpu': bento_args.cpu, 'memory': bento_args.memory},
  labels={'owner': 'bentoml-team', 'type': 'prebuilt'},
  image=bentoml.images.Image(python_version='3.11', lock_python_packages=False)
  .system_packages('libopenblas-dev', 'build-essential', 'pkg-config')
  .pyproject_toml('pyproject.toml'),
  envs=[
    {'name': 'UV_NO_PROGRESS', 'value': '1'},
    {'name': 'HF_HUB_DISABLE_PROGRESS_BARS', 'value': '1'},
    {'name': 'CMAKE_ARGS', 'value': '-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS'},
  ],
)
class LlamaCpp:
  model = bentoml.models.HuggingFaceModel(bento_args.model_id)

  @bentoml.on_startup
  def init_engine(self):
    self.llm = Llama.from_pretrained(repo_id=bento_args.model_id, filename=bento_args.filename, **bento_args.kwargs)

  @bentoml.api(route='/v1/chat/completions')
  async def chat_completions(
    self,
    messages: list[Message] = pydantic.Field(
      default=[{'role': 'user', 'content': 'Who are you? Please respond in priate speak!'}]
    ),
    functions: list[ChatCompletionFunction] | None = None,
    model: str = bento_args.model_id,
    max_tokens: t.Annotated[int, ae.Ge(128), ae.Le(bento_args.max_tokens)] = bento_args.max_tokens,
    temperature: float = bento_args.temperature,
    stop: list[str] | None = None,
    stream: bool = True,
    top_p: float | None = 1.0,
    frequency_penalty: float | None = 0.0,
  ):
    response = self.llm.create_chat_completion(
      model=model,
      messages=messages,
      max_tokens=max_tokens,
      stream=stream,
      stop=stop,
      temperature=temperature,
      top_p=top_p,
      frequency_penalty=frequency_penalty,
    )
    if stream:
      def streaming_response():
        for chunk in response:
          try:
            yield f'data: {json.dumps(chunk)}\n\n'
          except Exception:
            traceback.print_exc()
            yield 'data: Internal error. Check server logs\n\n'
            yield 'data: [DONE]\n\n'
            return
        yield 'data: [DONE]\n\n'
      return StreamingResponse(streaming_response(), media_type='text/event-stream')
    else:
      return JSONResponse(content=response)
