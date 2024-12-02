import base64
import json
import os
import time
from io import BytesIO
from typing import Optional, Union

from loguru import logger
from openai import NOT_GIVEN, NotGiven, OpenAI
from PIL.ImageFile import ImageFile

SYSTEM_PROMPT = """Analyze the provided text-image pair for sarcasm from both "sarcastic" and "non-sarcastic" perspectives, acknowledging the subjectivity in sarcasm recongnition.

Follow these steps for each perspective:

1. **Content Interpretation**: Understand the text and image separately.
2. **Interaction Analysis**: Identify any incongruities, irony, or exaggerations between the text and image.
3. **Contextual Evaluation**: Consider cultural or social contexts that may influence sarcasm.
4. **Confidence Scoring**: Assign an independent confidence score between 0 and 1 for the perspective, without the score being influenced by the other perspective.
5. **Explanation**: Provide a clear, confident, and reasonable explanation for the score from the respective perspective, explicitly stating "because..." to justify why it is or isn't sarcastic, limited to 5 sentences. Ensure that an explanation is always provided, even if the confidence level is low.
"""

RESPONSE_FORMAT = {
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
                    "description": "Explanation for sarcastic expression, justifying why it is sarcastic, limited to 5 sentences.",
                },
                "non_sarcastic_confidence": {
                    "type": "number",
                    "description": "Confidence score for non-sarcastic expression, ranging from 0 to 1.",
                },
                "non_sarcastic_explanation": {
                    "type": "string",
                    "description": "Explanation for non-sarcastic expression, justifying why it is non-sarcastic, limited to 5 sentences.",
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


def create_completion_param(
    text: str,
    image: ImageFile,
    model: str,
    temperature: float = 1.0,
    max_completion_tokens: Union[int, NotGiven] = 256,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
) -> dict:
    image_base64 = image_to_base64(image)
    image_data = f"data:image/jpeg;base64,{image_base64}"

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data}},
                {"type": "text", "text": text},
            ],
        },
    ]

    return {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_completion_tokens": max_completion_tokens,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "response_format": RESPONSE_FORMAT,
    }


def get_api_key_from_file(path: str, exclude_keys: Optional[list[str]] = None) -> str:
    success = False
    while not success:
        try:
            with open(path, "r") as f:
                keys: list[str] = json.load(f)
                success = True
        except Exception:
            logger.exception("Error in reading API key file [{}]", path)
            time.sleep(1 / 10)

    if exclude_keys is not None:
        new_keys = [key for key in keys if key not in exclude_keys]

        if len(new_keys) == 0:
            new_keys = keys

        return new_keys[0]

    return keys[0]


def get_base_url_from_file(path: str, model) -> str:
    while True:
        try:
            with open(path, "r") as f:
                keys: dict[str, str] = json.load(f)
                return keys[model]
        except Exception:
            logger.exception("Error in reading base url file [{}]", path)
            time.sleep(1 / 10)


def openai_requests_map_func(
    examples,
    api_keys_file_path: str,
    model: str = "gpt-4o",
    temperature: float = 1.0,
    max_completion_tokens: int = 256,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
) -> dict:

    api_key = get_api_key_from_file(api_keys_file_path)
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
        req = create_completion_param(
            text,
            image,
            model,
            temperature,
            max_completion_tokens if max_completion_tokens > 0 else NOT_GIVEN,
            top_p,
            frequency_penalty,
            presence_penalty,
        )

        success = False
        while not success:
            try:
                res = client.chat.completions.create(**req)
                res = res.to_json()
                if have_response:
                    examples[reponse_key_name][i] = res
                else:
                    examples[reponse_key_name].append(res)
                logger.success("{}-{}-{}", examples["id"][i], examples["label"][i], res)
                success = True
            except Exception:

                error_count += 1
                logger.exception("Error in request [{}]", examples["id"][i])
                time.sleep(60)

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

                api_key = get_api_key_from_file(api_keys_file_path, list(exclude_keys))
                client = OpenAI(api_key=api_key)
                continue

                # logger.error("Too many errors, skipping the rest")
                # examples[f"{model}_response"].append("")
                # success = True
                # error_count = 0
                # exclude_keys.clear()
                # api_key = get_api_key_from_file(api_key_file_path)
                # client = OpenAI(api_key=api_key)
                # continue

    return examples


def vllm_requests_map_func(
    examples,
    base_urls_file_path: str,
    model: str = "Qwen/Qwen2-VL-7B-Instruct",
    temperature: float = 1.0,
    max_completion_tokens: int = 256,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
) -> dict:

    base_url = get_base_url_from_file(base_urls_file_path, model)
    client = OpenAI(api_key="EMPTY", base_url=base_url)

    error_count = 0
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

        req = create_completion_param(
            text,
            image,
            model,
            temperature,
            max_completion_tokens if max_completion_tokens > 0 else NOT_GIVEN,
            top_p,
            frequency_penalty,
            presence_penalty,
        )

        success = False
        while not success:
            try:
                res = client.chat.completions.create(
                    **req, extra_body=dict(guided_decoding_backend="outlines")
                )
                res = res.to_json()
                if have_response:
                    examples[reponse_key_name][i] = res
                else:
                    examples[reponse_key_name].append(res)
                logger.success("{}-{}-{}", examples["id"][i], examples["label"][i], res)
                success = True
            except Exception:

                error_count += 1
                logger.exception("Error in request [{}]", examples["id"][i])
                time.sleep(60)

                if error_count <= 10:
                    error_count += 1
                    continue

                # if error_count > 10:
                logger.warning("Too many errors, changing base url")
                base_url = get_base_url_from_file(base_urls_file_path, model)
                client = OpenAI(api_key="EMPTY", base_url=base_url)
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
