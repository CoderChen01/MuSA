import base64
import json
import os
import time
from io import BytesIO
from typing import Any, Optional, Union

from loguru import logger
from openai import NOT_GIVEN, LengthFinishReasonError, NotGiven, OpenAI
from PIL.ImageFile import ImageFile

SYSTEM_PROMPT = (
    lambda x: f"""Analyze the provided text-image pair from both sarcastic and non-sarcastic perspectives, acknowledging the inherent subjectivity in sarcasm recognition.

For each perspective, follow these steps independently:

### 1. **Sarcastic Perspective**

1. **Content Interpretation**: Analyze the text and image separately to understand the content, with a focus on potential irony, exaggeration, or contradictions within the text and image.
2. **Interaction Analysis**: Identify any incongruities, irony, or exaggerations between the text and the image that may suggest sarcasm. Look for contrasts or mismatches where the text sharply deviates from the image.
3. **Contextual Evaluation**: Consider any cultural, social, or situational factors that might influence whether the text is interpreted as sarcastic, drawing from common sarcasm markers or humor conventions.
4. **Confidence Scoring**: Assign a confidence score between 0 and 1 for the sarcastic perspective, **independent of the non-sarcastic score**. The score should reflect how strongly the text-image pair suggests sarcasm.
5. **Explanation**: Provide a clear and confident explanation for the sarcastic score. The explanation should justify the score by clearly stating why the pair is sarcastic, with reasoning that flows logically. Ensure the explanation is concise (limit to {x} sentences) and assertive, even if the score is low.

### 2. **Non-Sarcastic Perspective**

1. **Content Interpretation**: Analyze the text and image separately, focusing on how the text could be understood literally, or without irony, in relation to the image.
2. **Interaction Analysis**: Evaluate whether the text and image align in a way that supports a non-ironic or straightforward interpretation. Look for direct relationships between the text and the image without contradiction or exaggeration.
3. **Contextual Evaluation**: Consider cultural, social, or situational contexts where the text might be interpreted in a non-sarcastic manner, such as simple observations or positive expressions that are consistent with the image.
4. **Confidence Scoring**: Assign a confidence score between 0 and 1 for the non-sarcastic perspective, **independent of the sarcastic score**. The score should reflect how likely the text-image pair is interpreted in a non-sarcastic way.
5. **Explanation**: Provide a clear and confident explanation for the non-sarcastic score. Justify why the pair is not sarcastic by explaining the reasoning in a cause-and-effect manner. The explanation should be concise (limit to {x} sentences), confident, and assertive, even if the confidence is low.

**Note**: The confidence scores for `sarcastic` and `non-sarcastic` are independent and do not need to add up to 1. Both perspectives should be evaluated and scored independently.
"""
)

RESPONSE_FORMAT = lambda x: {
    "type": "json_schema",
    "json_schema": {
        "name": "multimodal_sarcasm_analysis_from_multiple_perspectives",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "sarcastic_confidence": {
                    "type": "number",
                    "description": "Confidence score for sarcastic expression, ranging from 0 to 1.",
                },
                "sarcastic_explanation": {
                    "type": "string",
                    "description": f"Explanation for sarcastic expression, justifying why it is sarcastic, limited to {x} sentences.",
                },
                "non_sarcastic_confidence": {
                    "type": "number",
                    "description": "Confidence score for non-sarcastic expression, ranging from 0 to 1.",
                },
                "non_sarcastic_explanation": {
                    "type": "string",
                    "description": f"Explanation for non-sarcastic expression, justifying why it is non-sarcastic, limited to {x} sentences.",
                },
            },
            "required": [
                "sarcastic_confidence",
                "sarcastic_explanation",
                "non_sarcastic_confidence",
                "non_sarcastic_explanation",
            ],
            "additionalProperties": False,
        },
    },
}


def image_to_base64(image: ImageFile) -> str:
    with BytesIO() as buffered:
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


class CompletionParamBuilder:

    soft_num_sentences_limit_prompts = [
        "very short",
        "a few",
        "several",
        "minimal",
        "very compact",
        "vary succinct",
        "only a handlful of",
    ]

    def __init__(self, seed: int) -> None:
        self._temperature = 0.0
        self._seed = seed

        self._num_sentences = 5

        self._soft_prompt = self._num_sentences

    def __call__(
        self,
        text: str,
        image: ImageFile,
        model: str,
    ) -> dict:
        image_base64 = image_to_base64(image)
        image_data = f"data:image/jpeg;base64,{image_base64}"

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT(self._soft_prompt),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data}},
                    {"type": "text", "text": text},
                ],
            },
        ]
        response_format = RESPONSE_FORMAT(self._soft_prompt)

        self._num_sentences -= 1
        if self._num_sentences <= 0:
            if len(self.soft_num_sentences_limit_prompts) <= abs(
                self._num_sentences
            ):  # If the length is never enough, it will return empty directly and count it as an exception.
                self._soft_prompt = 5
                self._temperature += 0.1
                if self._temperature >= 1.0:
                    self._temperature = 1.0
                if self._temperature == 1.0:
                    self._seed += 10
            else:
                self._soft_prompt = self.soft_num_sentences_limit_prompts[
                    self._num_sentences
                ]
        else:
            self._soft_prompt = self._num_sentences

        return {
            "model": model,
            "messages": messages,
            "temperature": self._temperature,
            "max_completion_tokens": NOT_GIVEN,
            "top_p": 1.0,
            "seed": NOT_GIVEN if self._temperature == 0.0 else self._seed,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "response_format": response_format,
        }


def get_api_key(keys: list[str], exclude_keys: Optional[list[str]] = None) -> str:

    if exclude_keys is not None:
        new_keys = [key for key in keys if key not in exclude_keys]

        if len(new_keys) == 0:
            new_keys = keys

        return new_keys[0]

    return keys[0]


def get_base_url(
    base_urls: dict[str, list[str]],
    model: str,
    excluded_base_urls: Optional[list[str]] = None,
) -> str:
    urls = base_urls[model]

    if excluded_base_urls is not None:
        new_urls = [url for url in urls if url not in excluded_base_urls]

        if len(new_urls) == 0:
            new_urls = urls

        return new_urls[0]

    return urls[0]


def get_config(path: str, key: str, **kwargs) -> Any:

    success_load = False
    while not success_load:
        try:
            with open(path, "r") as f:
                data = json.load(f)
            success_load = True
        except Exception:
            logger.exception("Error in reading config file [{}]", path)
            time.sleep(1 / 10)

    if key == "api_keys":
        keys: list[str] = data[key]
        return get_api_key(keys, **kwargs)
    elif key == "base_urls":
        base_urls: dict[str, list[str]] = data[key]
        return get_base_url(base_urls, **kwargs)

    return data[key]


def openai_requests_map_func(
    examples,
    config_file_path: str,
    model: str = "gpt-4o",
    seed: int = 42,
) -> dict:

    api_key = get_config(config_file_path, "api_keys")
    client = OpenAI(api_key=api_key)

    error_count = 0
    exclude_keys = set()
    have_response = False

    reponse_key_name = f"{model}_MuSA"

    if reponse_key_name not in examples:
        examples[reponse_key_name] = []
    else:
        have_response = True

    for i in range(len(examples["text"])):
        if have_response and examples[reponse_key_name][i] != "":
            logger.info("Already have response for [{}]", examples["id"][i])
            continue

        image = examples["image"][i]
        text = examples["text"][i]

        req_builder = CompletionParamBuilder(seed)
        req = req_builder(text, image, model)

        is_request = True
        success = False
        while not success:
            try:
                if is_request:
                    with client.beta.chat.completions.stream(**req) as st:
                        res = st.get_final_completion()
                    res = res.to_json()
                    if have_response:
                        examples[reponse_key_name][i] = res
                    else:
                        examples[reponse_key_name].append(res)
                    logger.success(
                        "{}-{}-{}", examples["id"][i], examples["label"][i], res
                    )
                    success = True
                else:
                    is_request = get_config(config_file_path, "metadata")["is_request"]
                    logger.info("Request paused for [{}]", examples["id"][i])
                    time.sleep(10)
            except LengthFinishReasonError:
                logger.warning("Length finish reason error for [{}]", examples["id"][i])
                req = req_builder(text, image, model)
                logger.info("Retry with [{}]", req)
            except Exception:
                is_request = get_config(config_file_path, "metadata")["is_request"]

                error_count += 1
                logger.exception("Error in request [{}]", examples["id"][i])
                time.sleep(10)

                if error_count <= 10:
                    error_count += 1
                    continue

                # if error_count > 10:
                logger.warning("Too many errors, changing API key")
                old_exclude_key_len = len(exclude_keys)
                exclude_keys.add(api_key)
                if old_exclude_key_len == len(exclude_keys):
                    # error_count += 1
                    exclude_keys.clear()

                api_key = get_config(
                    config_file_path, "api_keys", exclude_keys=list(exclude_keys)
                )
                client = OpenAI(api_key=api_key)
                continue

    return examples


def vllm_requests_map_func(
    examples,
    config_file_path: str,
    model: str = "Qwen/Qwen2-VL-7B-Instruct",
    seed: int = 42,
) -> dict:

    base_url = get_config(config_file_path, "base_urls", model=model)
    client = OpenAI(api_key="EMPTY", base_url=base_url, max_retries=0)

    error_count = 0
    excluded_base_urls = set()
    have_response = False

    reponse_key_name = f"{model.replace('/', '--')}_MuSA"

    if reponse_key_name not in examples:
        examples[reponse_key_name] = []
    else:
        have_response = True

    for i in range(len(examples["text"])):
        if have_response and examples[reponse_key_name][i] != "":
            logger.info("Already have response for [{}]", examples["id"][i])
            continue

        image = examples["image"][i]
        text = examples["text"][i]

        req_builder = CompletionParamBuilder(seed)
        req = req_builder(text, image, model)

        is_request = True
        success = False
        while not success:
            try:
                if is_request:
                    with client.beta.chat.completions.stream(
                        **req, extra_body=dict(guided_decoding_backend="outlines")
                    ) as st:
                        res = st.get_final_completion()
                    res = res.to_json()
                    if have_response:
                        examples[reponse_key_name][i] = res
                    else:
                        examples[reponse_key_name].append(res)
                    logger.success(
                        "{}-{}-{}", examples["id"][i], examples["label"][i], res
                    )
                    success = True
                else:
                    is_request = get_config(config_file_path, "metadata")["is_request"]
                    logger.info("Request paused for [{}]", examples["id"][i])
                    time.sleep(10)
            except LengthFinishReasonError:
                logger.warning("Length finish reason error for [{}]", examples["id"][i])
                req = req_builder(text, image, model)
                logger.info("Retry with [{}]", req)
            except Exception:
                is_request = get_config(config_file_path, "metadata")["is_request"]

                error_count += 1
                logger.exception("Error in request [{}]", examples["id"][i])
                time.sleep(10)

                if error_count <= 10:
                    error_count += 1
                    continue

                logger.warning("Too many errors, change base url")
                old_exclude_base_urls_len = len(excluded_base_urls)
                excluded_base_urls.add(base_url)
                if old_exclude_base_urls_len == len(excluded_base_urls):
                    # error_count += 1
                    excluded_base_urls.clear()

                # if error_count > 10:
                base_url = get_config(
                    config_file_path,
                    "base_urls",
                    model=model,
                    excluded_base_urls=list(excluded_base_urls),
                )
                client = OpenAI(api_key="EMPTY", base_url=base_url, max_retries=0)
                continue

    return examples


def local_data_map_func(
    examples,
    models: list[str],
    local_data: Optional[dict[str, dict]] = None,
) -> dict:

    for model in models:
        have_response = False
        model_name = model.replace("/", "--")
        reponse_key_name = f"{model_name}_MuSA"

        if reponse_key_name not in examples:
            examples[reponse_key_name] = []
        else:
            have_response = True

        for i in range(len(examples["text"])):
            if have_response and examples[reponse_key_name][i] != "":
                logger.info("Already have response for [{}]", examples["id"][i])
                continue

            id_ = f"{examples['id'][i]}-{model_name}"

            if local_data is not None and id_ in local_data:
                if have_response:
                    examples[reponse_key_name][i] = json.dumps(local_data[id_])
                else:
                    examples[reponse_key_name].append(json.dumps(local_data[id_]))
                logger.info("Using local data for [{}]", id_)
                continue

            logger.info("No local data for [{}]", id_)
            if not have_response:
                examples[reponse_key_name].append("")

    return examples
