import requests

try:
    from langchain_core.language_models.llms import BaseLLM
    from langchain_core.outputs import Generation, LLMResult
    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
    from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
except ImportError as e:
    print(
        "ModelArena core llms is wrapped over LangChain LLMs."
        "To use it, you have to install langchain by:\npip install langchain\n"
    )
    raise e


class BytedChatGPT(BaseLLM):
    jwt_url: str = "https://cloud.bytedance.net/auth/api/v1/jwt"
    byted_gpt_url: str = "https://chatverse.bytedance.net/v1/chat"

    model: str = Field(default="gpt-35-turbo", alias="model_name")
    temperature: float = 0.2
    top_p: float = 0.95
    timeout: int = 60
    model_kwargs: dict[str, object] = Field(default_factory=dict)

    prefix_messages: list = Field(default_factory=list)

    byted_gpt_token: SecretStr | None = None

    @root_validator()
    def validate_environment(cls, values: dict[str, object]) -> dict[str, object]:
        values["byted_gpt_token"] = convert_to_secret_str(
            get_from_dict_or_env(values, "byted_gpt_token", "BYTED_GPT_TOKEN")
        )
        return values

    @property
    def _default_params(self) -> dict[str, object]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            **self.model_kwargs,
        }

    @property
    def _llm_type(self) -> str:
        return "byted-chatgpt"

    def _create(
        self,
        messages: list[dict[str, str]],
        params: dict[str, object],
    ) -> dict[str, object]:
        jwt_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.byted_gpt_token.get_secret_value()}",
        }
        try:
            jwt_response = requests.get(self.jwt_url, headers=jwt_headers, timeout=self.timeout)
            jwt_headers = jwt_response.headers
        except Exception as e:
            raise ValueError(f"An error has occurred to get jwt header: {e}")

        params.update({"messages": messages})
        try:
            response = requests.post(self.byted_gpt_url, headers=jwt_headers, json=params, timeout=self.timeout)
            if response.status_code == 200:
                parsed_json = response.json()
                if parsed_json.get("code") == 0:
                    return parsed_json["data"]
                else:
                    raise ValueError(f"An error has occurred to get reponse: {parsed_json['message']}")
            else:
                response.raise_for_status()
        except Exception as e:
            raise ValueError(f"An error has occurred to get response: {e}")

    def _get_chat_params(
        self,
        prompts: list[str],
        stop: list[str] | None = None,
    ) -> tuple[list[dict[str, str]], dict[str, object]]:
        # this function comes from langchain implementation of OpenAI chat models
        if len(prompts) > 1:
            raise ValueError(f"OpenAIChat currently only supports single prompt, got {prompts}")

        messages = self.prefix_messages + [{"role": "user", "content": prompts[0]}]
        params: dict[str, object] = {**{"model": self.model}, **self._default_params}

        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop

        if params.get("max_tokens") == -1:
            # for ChatGPT api, omitting max_tokens is equivalent to having no limit
            del params["max_tokens"]

        return messages, params

    def _generate(
        self,
        prompts: list[str],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: dict[str, object],
    ) -> LLMResult:
        messages, params = self._get_chat_params(prompts, stop)
        params = {**params, **kwargs}

        response = self._create(messages, params)
        llm_output = {
            "token_usage": response["usage"],
            "model_name": self.model,
        }
        return LLMResult(
            generations=[
                [Generation(text=response["choices"][0]["message"]["content"])],
            ],
            llm_output=llm_output,
        )
