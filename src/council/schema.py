# ABOUTME: Shared output schema for the LLM council.
# ABOUTME: All providers must emit {label, rationale, confidence} with these enums.

from typing import Literal

LiberyLabel = Literal["positive_liberty", "negative_liberty", "ambiguous", "other"]
LABEL_VALUES: tuple[str, ...] = ("positive_liberty", "negative_liberty", "ambiguous", "other")


# JSON Schema for structured output — used to constrain GPT (json_schema response_format)
# and Gemini (response_schema). Claude tool-use uses the same shape via input_schema.
COUNCIL_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "rationale": {
            "type": "string",
            "description": (
                "One sentence (max 200 chars). Name the liberty object; state whether the "
                "sentence concerns non-interference, enabled capacity, ambiguity, or "
                "non-substantive usage."
            ),
        },
        "label": {
            "type": "string",
            "enum": list(LABEL_VALUES),
        },
        "confidence": {
            "type": "number",
            "description": (
                "Self-reported confidence in this label, a number between 0.0 and 1.0. "
                "0.6 = roughly two-thirds sure; 0.9 = the sentence is unambiguous. "
                "Values outside [0,1] will be clamped on read."
            ),
        },
    },
    "required": ["rationale", "label", "confidence"],
    "additionalProperties": False,
}


# Tool definition for Claude (forced via tool_choice). Same shape, different wrapper.
CLAUDE_TOOL = {
    "name": "classify_liberty",
    "description": "Record the classification of the sentence according to Berlin's distinction.",
    "strict": True,
    "input_schema": COUNCIL_OUTPUT_SCHEMA,
}
