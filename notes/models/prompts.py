"""
Methods for processing prompts used in LLM API requests.
"""
import logging
from typing import Dict, List, Union

from notes.config.settings import Settings

# import readmeai.config.settings as Settings
# from readmeai.core.logger import Logger

_logger = logging.getLogger(__name__)


def get_prompt_context(prompts: dict, prompt_type: str, context: dict) -> str:
    """Generates a prompt for the LLM API."""
    prompt_template = get_prompt_template(prompts, prompt_type)

    if not prompt_template:
        _logger.error(f"Prompt type '{prompt_type}' not found.")
        return ""

    return inject_prompt_context(prompt_template, context)


def get_prompt_template(prompts: dict, prompt_type: str) -> str:
    """Retrieves the template for the given prompt type."""
    prompt_templates = {
        "features": prompts["prompts"]["features"],
        "overview": prompts["prompts"]["overview"],
        "slogan": prompts["prompts"]["slogan"],
    }
    return prompt_templates.get(prompt_type, "")


def inject_prompt_context(template: str, context: dict) -> str:
    """Formats the template with the provided context."""
    try:
        return template.format(*[context[key] for key in context])
    except KeyError as exc:
        _logger.error(f"Missing context for prompt key: {exc}")
        return ""


async def set_additional_contexts(
    config: Settings,
    dependencies: List[str],
    file_summaries: List[str],
) -> List[dict]:
    """Generates additional prompts (features, overview, slogan) for LLM."""
    return [
        {"type": prompt_type, "context": context}
        for prompt_type, context in [
            (
                "features",
                {
                    "repo": config.git.repository,
                    "dependencies": dependencies,
                    "file_summary": file_summaries,
                },
            ),
            (
                "overview",
                {
                    "name": config.git.name,
                    "file_summary": file_summaries,
                },
            ),
            (
                "slogan",
                {
                    "name": config.git.name,
                    "repo": config.git.repository,
                    "file_summary": file_summaries,
                },
            ),
        ]
    ]


async def set_summary_context(
    config: Settings,
    dependencies: List[str],
    file_summaries: List[str],
) -> List[Dict[str, Union[str, dict]]]:
    """Generates the summary prompts to be used by the LLM API."""
    return [
        {"type": prompt_type, "context": context}
        for prompt_type, context in [
            (
                "file_summary",
                {
                    "tree": config.md.tree,
                    "dependencies": dependencies,
                    "file_summary": file_summaries,
                },
            ),
        ]
    ]
