"""Reference-action agreement: valid format, action type, then its parameters.

These are offline next-action metrics, not browser-task success. Position and
scroll errors use sqrt((dx / width)^2 + (dy / height)^2). Text similarity is
SequenceMatcher's character-sequence ratio, not semantic or edit-distance credit.
"""

import ast
from collections import defaultdict
from difflib import SequenceMatcher
import math


ACTION_ARGUMENTS = {
    "mouse_click": {"x": float, "y": float, "button": str},
    "mouse_drag_and_drop": {"from_x": float, "from_y": float, "to_x": float, "to_y": float},
    "keyboard_type": {"text": str},
    "keyboard_press": {"key": str},
    "scroll": {"delta_x": float, "delta_y": float},
    "scroll_at": {"x": float, "y": float, "delta_x": float, "delta_y": float},
    "goto": {"url": str},
    "new_tab": {},
    "send_msg_to_user": {"text": str},
}
DEFAULT_SETTINGS = {
    "position_tolerance": 0.05,
    "scroll_tolerance": 0.25,
    "scroll_deadzone_pixels": 1.0,
}
ACTION_ALIASES = {
    "click": "mouse_click", "type": "keyboard_type",
    "keypress": "keyboard_press", "drag_and_drop": "mouse_drag_and_drop",
}


def _read_action_call(text):
    try:
        call = ast.parse(text.strip(), mode="eval").body
    except (SyntaxError, TypeError, OverflowError) as error:
        raise ValueError("Invalid action syntax or arguments.") from error
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        raise ValueError("Expected a supported function call.")
    return call, ACTION_ALIASES.get(call.func.id, call.func.id)


def _action_parameters(call, name):
    if name not in ACTION_ARGUMENTS:
        raise ValueError(f"Unsupported action: {name}")
    argument_types = ACTION_ARGUMENTS[name]
    if len(call.args) > len(argument_types):
        raise ValueError(f"Wrong arguments for {name}.")
    try:
        parameters = {key: ast.literal_eval(value) for key, value in zip(argument_types, call.args)}
        for keyword in call.keywords:
            if keyword.arg in parameters or keyword.arg not in argument_types:
                raise ValueError(f"Wrong arguments for {name}.")
            parameters[keyword.arg] = ast.literal_eval(keyword.value)
        if name == "mouse_click":
            parameters.setdefault("button", "left")
        if parameters.keys() != argument_types.keys():
            raise ValueError(f"Wrong arguments for {name}.")
        for key, expected in argument_types.items():
            value = parameters[key]
            if expected is float:
                valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            else:
                valid = isinstance(value, str)
            if not valid:
                raise ValueError(f"Wrong argument type for {name}.{key}.")
        if name == "mouse_click" and parameters["button"] not in {"left", "middle", "right"}:
            raise ValueError("Unsupported mouse button.")
        return parameters
    except (SyntaxError, TypeError, OverflowError) as error:
        raise ValueError("Invalid action syntax or arguments.") from error


def parse_action(text):
    """Read literal arguments from a supported function call without executing it."""
    call, name = _read_action_call(text)
    return {"type": name, "parameters": _action_parameters(call, name)}


def _point_metrics(predicted, target, width, height, tolerance, prefix="position"):
    dx, dy = predicted[0] - target[0], predicted[1] - target[1]
    normalized = math.hypot(dx / width, dy / height)
    return {
        f"{prefix}_error_pixels": math.hypot(dx, dy),
        f"{prefix}_error_normalized": normalized,
        f"{prefix}_match": normalized <= tolerance,
    }


def _text_metrics(predicted, target):
    return {
        "text_exact_match": predicted == target,
        "text_similarity": SequenceMatcher(None, predicted, target, autojunk=False).ratio(),
    }


def _message_kind(text):
    if text.startswith("[EXIT]"):
        return "exit"
    if text.startswith("[ANSWER]"):
        return "answer"
    return "message"


def _direction(value, deadzone):
    return 0 if abs(value) <= deadzone else (1 if value > 0 else -1)


def score_action(prediction, target, width, height, settings=None):
    """Score one reference action. Invalid predictions remain failures.

    Parameter fields exist only when the supported action type matches. The
    unconditional joint_match prevents those conditional metrics hiding errors.
    A malformed reference raises ValueError instead of silently reducing support.
    """
    if width <= 0 or height <= 0:
        raise ValueError("Screenshot dimensions must be positive.")
    settings = {**DEFAULT_SETTINGS, **(settings or {})}
    reference = parse_action(target)
    reference_type = reference["type"]
    result = {
        "target_type": reference_type,
        "predicted_type": None,
        "parse_valid": False,
        "type_match": False,
        "parameters_valid": False,
        "complete_match": False,
        "joint_match": False,
    }
    try:
        call, name = _read_action_call(prediction)
        # A recognized call can earn type credit even with invalid arguments.
        result.update(predicted_type=name, type_match=name == reference_type)
        predicted = _action_parameters(call, name)
    except ValueError as error:
        result["parse_error"] = str(error)
        return result
    result.update(parse_valid=True, parameters_valid=True)
    if name != reference_type:
        return result
    expected = reference["parameters"]
    result["complete_match"] = predicted == expected
    tolerance = settings["position_tolerance"]

    if name in {"mouse_click", "scroll_at"}:
        result.update(_point_metrics(
            (predicted["x"], predicted["y"]), (expected["x"], expected["y"]), width, height, tolerance
        ))
    if name == "mouse_click":
        result["button_match"] = predicted["button"] == expected["button"]
        result["joint_match"] = result["position_match"] and result["button_match"]
    elif name == "mouse_drag_and_drop":
        for prefix in ("from", "to"):
            result.update(_point_metrics(
                (predicted[f"{prefix}_x"], predicted[f"{prefix}_y"]),
                (expected[f"{prefix}_x"], expected[f"{prefix}_y"]),
                width, height, tolerance, prefix,
            ))
        result["joint_match"] = result["from_match"] and result["to_match"]
    elif name in {"scroll", "scroll_at"}:
        deadzone = settings["scroll_deadzone_pixels"]
        predicted_vector = predicted["delta_x"] / width, predicted["delta_y"] / height
        expected_vector = expected["delta_x"] / width, expected["delta_y"] / height
        result["scroll_direction_match"] = all(
            _direction(predicted[key], deadzone) == _direction(expected[key], deadzone)
            for key in ("delta_x", "delta_y")
        )
        result["scroll_vector_error_normalized"] = math.hypot(
            predicted_vector[0] - expected_vector[0], predicted_vector[1] - expected_vector[1]
        )
        result["scroll_magnitude_error_normalized"] = abs(
            math.hypot(*predicted_vector) - math.hypot(*expected_vector)
        )
        result["scroll_amount_match"] = result["scroll_magnitude_error_normalized"] <= settings["scroll_tolerance"]
        result["scroll_vector_match"] = result["scroll_vector_error_normalized"] <= settings["scroll_tolerance"]
        result["joint_match"] = result["scroll_direction_match"] and result["scroll_vector_match"]
        if name == "scroll_at":
            result["joint_match"] = result["joint_match"] and result["position_match"]
    elif name == "keyboard_press":
        # Preserve keys such as ControlOrMeta+a; do not assume platform aliases.
        result["key_match"] = predicted["key"] == expected["key"]
        result["joint_match"] = result["key_match"]
    elif name in {"keyboard_type", "send_msg_to_user"}:
        result.update(_text_metrics(predicted["text"], expected["text"]))
        if name == "send_msg_to_user":
            result["target_message_kind"] = _message_kind(expected["text"])
            result["predicted_message_kind"] = _message_kind(predicted["text"])
            result["message_kind_match"] = result["target_message_kind"] == result["predicted_message_kind"]
        result["joint_match"] = result["text_exact_match"]
    elif name == "goto":
        # Fragments and query strings can change browser state, so retain them.
        result["url_match"] = predicted["url"] == expected["url"]
        result["joint_match"] = result["url_match"]
    elif name == "new_tab":
        result["joint_match"] = True
    return result


def _mean(values):
    return sum(values) / len(values) if values else None


def _parameter_summary(scores):
    excluded = {"parse_valid", "type_match", "complete_match", "joint_match"}
    values = defaultdict(list)
    for score in scores:
        if score["type_match"]:
            for key, value in score.items():
                if key not in excluded and isinstance(value, (bool, int, float)):
                    values[key].append(value)
    return {key: {"mean": _mean(items), "count": len(items)} for key, items in values.items()}


def _score_group(scores):
    # Accuracy denominators include every draw in this group.
    return {
        "count": len(scores),
        "type_accuracy": _mean([score["type_match"] for score in scores]),
        "joint_match_rate": _mean([score["joint_match"] for score in scores]),
        "parameters_given_type_match": _parameter_summary(scores),
    }


def aggregate_scores(scores):
    """Report unconditional agreement plus conditional errors with their support."""
    scores = list(scores)
    groups = defaultdict(list)
    for score in scores:
        groups[score["target_type"]].append(score)
    per_type = {name: _score_group(items) for name, items in groups.items()}
    return {
        **_score_group(scores),
        "parse_valid_rate": _mean([score["parse_valid"] for score in scores]),
        "macro_type_accuracy": _mean([item["type_accuracy"] for item in per_type.values()]),
        "complete_match_rate": _mean([score["complete_match"] for score in scores]),
        "type_match_count": sum(score["type_match"] for score in scores),
        "per_type": per_type,
    }
