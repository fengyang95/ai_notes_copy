from tqdm.auto import tqdm
from abc import ABC, abstractmethod

from pandas.core.frame import DataFrame

from .llms import BytedChatGPT

try:
    import torch
except ImportError as e:
    print(
        "ModelArena core engine depends on PyTorch package."
        "To use it, you have to install pytorch by:\npip install torch\n"
    )
    raise e

try:
    from langchain_core.language_models.llms import BaseLLM
    from langchain_community.llms import VLLM
    from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
except ImportError as e:
    print(
        "ModelArena core engine is wrapped over LangChain LLMs."
        "To use it, you have to install langchain by:\npip install langchain\n"
    )
    raise e


class LLMEngine(ABC):
    engine_name: str = "{model}"
    engine_method: str

    model: str
    model_path: str
    llm: BaseLLM
    generation_kwargs: dict[str, object]

    show_progress: bool = False

    @abstractmethod
    def _init_engine(self) -> None:
        ...

    def __init__(
        self,
        model: str,
        model_path: str,
        generation_kwargs: dict[str, object] = ...,
        show_progress: bool = False,
    ) -> None:
        self.model = model
        self.model_path = model_path
        self.generation_kwargs = generation_kwargs
        self.show_progress = show_progress

        self.engine_name = self.engine_name.format(model=model)
        self._init_engine()

        tqdm.pandas(disable=not self.show_progress)

    @abstractmethod
    def _infer(self, inputs: str) -> str:
        ...

    def infer(self, df: DataFrame) -> DataFrame:
        df["output"] = df["prompt"].progress_apply(self._infer)
        return df


class HuggingFaceEngine(LLMEngine):
    engine_method: str = "huggingface"

    def _init_engine(self) -> None:
        self.llm = HuggingFacePipeline.from_model_id(
            model_id=self.model_path,
            task="text-generation",
            device=0 if torch.cuda.is_available() else -1,
            pipeline_kwargs=self.generation_kwargs,
        )

    def _infer(self, inputs: str) -> str:
        outputs = self.llm.invoke(inputs).replace(inputs, "")
        return outputs


class VLLMEngine(LLMEngine):
    engine_method: str = "vllm"

    def _init_engine(self) -> None:
        self.llm = VLLM(
            model=self.model_path,
            trust_remote_code=True,
            **self.generation_kwargs,
        )

    def _infer(self, inputs: str) -> str:
        outputs = self.llm.invoke(inputs)
        return outputs


class BytedChatGPTEngine(LLMEngine):
    engine_name: str = "byted_chatgpt"

    def _init_engine(self) -> None:
        self.llm = BytedChatGPT(
            model=self.model,
            **self.generation_kwargs,
        )

    def _infer(self, inputs: str) -> str:
        outputs = self.llm.invoke(inputs)
        return outputs
