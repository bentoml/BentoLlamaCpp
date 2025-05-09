from __future__ import annotations

import os, bentoml, pydantic, fastapi, traceback, typing as t, annotated_types as ae
import time
import uuid

from starlette.responses import JSONResponse, StreamingResponse

# with bentoml.importing():
#   from llama_cpp import Llama, ChatCompletionFunction


class Message(pydantic.BaseModel):
  role: t.Literal['system', 'user', 'assistant']
  content: str


# These models are to mock the behavior of llama-cpp-python's response objects
class MockDelta(pydantic.BaseModel):
    role: t.Optional[str] = None
    content: t.Optional[str] = None

class MockChoiceChunk(pydantic.BaseModel):
    index: int
    delta: MockDelta
    logprobs: t.Optional[t.Any] = None
    finish_reason: t.Optional[str] = None

class MockStreamingCompletionResponse(pydantic.BaseModel):
    id: str = pydantic.Field(default_factory=lambda: f"chatcmpl-mock-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion.chunk"
    created: int = pydantic.Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[MockChoiceChunk]

class MockMessageResponse(pydantic.BaseModel):
    role: str
    content: str

class MockChoice(pydantic.BaseModel):
    index: int
    message: MockMessageResponse
    logprobs: t.Optional[t.Any] = None
    finish_reason: t.Optional[str] = None

class MockUsage(pydantic.BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class MockNonStreamingCompletionResponse(pydantic.BaseModel):
    id: str = pydantic.Field(default_factory=lambda: f"chatcmpl-mock-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion"
    created: int = pydantic.Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[MockChoice]
    usage: MockUsage

class MockLlama:
    def __init__(self, model_id: str):
        self.model_id = model_id
        print(f"Initialized MockLlama with model_id: {self.model_id}")

    def create_chat_completion_openai_v1(
        self,
        model: str,
        messages: list[Message],
        max_tokens: int,
        stream: bool,
        stop: t.Optional[list[str]] = None,
        temperature: float = 0.7,
        top_p: t.Optional[float] = 1.0,
        frequency_penalty: t.Optional[float] = 0.0,
        # presence_penalty: t.Optional[float] = 0.0, # common OpenAI param
        # user: t.Optional[str] = None, # common OpenAI param
    ):
        print(f"MockLlama.create_chat_completion_openai_v1 called with model: {model}, stream: {stream}")
        # Use the 'model' argument passed to this method, which originates from the request or bento_args.model_id

        if stream:
            # Simulate streaming response token by token
            response_tokens = ['', 'Ah', 'oy', '!', ' This', ' be', ' a', ' mock', ' non', '-stream', 'ing', ' response', ',', ' savvy', '?']

            # First chunk: send the role
            yield MockStreamingCompletionResponse(
                model=model,
                choices=[MockChoiceChunk(index=0, delta=MockDelta(role="assistant"))]
            )

            # Subsequent chunks: send content tokens
            for token in response_tokens:
                if token: # Don't send empty string content if it's the first token in list
                    yield MockStreamingCompletionResponse(
                        model=model,
                        choices=[MockChoiceChunk(index=0, delta=MockDelta(content=token))]
                    )

            # Final chunk: send finish reason
            yield MockStreamingCompletionResponse(
                model=model,
                choices=[MockChoiceChunk(index=0, delta=MockDelta(), finish_reason="stop")],
            )
        else:
            # Simulate non-streaming response
            return MockNonStreamingCompletionResponse(
                model=model,
                choices=[
                    MockChoice(
                        index=0,
                        message=MockMessageResponse(role="assistant", content="Ahoy! This be a mock non-streaming response, savvy?"),
                        finish_reason="stop"
                    )
                ],
                usage=MockUsage(prompt_tokens=sum(len(m.content) for m in messages), completion_tokens=15, total_tokens=sum(len(m.content) for m in messages) + 15)
            )

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
openai_api_app = fastapi.FastAPI(redoc_url=None, docs_url=None)


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
  @bentoml.on_startup
  def init_engine(self):
    # self.llm = Llama.from_pretrained(repo_id=bento_args.model_id, filename=bento_args.filename, **bento_args.kwargs)
    self.llm = MockLlama(model_id=bento_args.model_id)

  @bentoml.api(route='/v1/chat/completions')
  async def chat_completions(
    self,
    messages: list[Message] = pydantic.Field(
      default=[Message(role='user', content='Who are you? Please respond in pirate speak!')]
    ),
    functions: list[t.Any] | None = None,
    model: str = bento_args.model_id,
    max_tokens: t.Annotated[int, ae.Ge(128), ae.Le(bento_args.max_tokens)] = bento_args.max_tokens,
    temperature: float = bento_args.temperature,
    stop: list[str] | None = None,
    stream: bool = False,
    top_p: float | None = 1.0,
    frequency_penalty: float | None = 0.0,
  ):
    response = self.llm.create_chat_completion_openai_v1(
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
            yield f'data: {chunk.model_dump_json()}\n\n'
          except Exception:
            traceback.print_exc()
            yield 'data: Internal error. Check server logs\n\n'
            yield 'data: [DONE]\n\n'
            return
        yield 'data: [DONE]\n\n'

      return StreamingResponse(streaming_response(), media_type='text/event-stream')
    else:
      return JSONResponse(content=response.model_dump())
