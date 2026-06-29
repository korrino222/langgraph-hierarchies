"""State reducer functions for BaseState fields."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from uuid import uuid4

from langchain_core.messages import BaseMessage
from langchain_core.messages.utils import convert_to_messages

if TYPE_CHECKING:
    from langgraph_hierarchies.types import Progress

logger = logging.getLogger(__name__)


def reduce_messages(
    left: list[BaseMessage],
    right: list[BaseMessage],
) -> list[BaseMessage]:
    """Merge messages by ID, replacing matches and appending new ones."""
    if any(isinstance(message, dict) for message in left):
        left = convert_to_messages(left)
    if any(isinstance(message, dict) for message in right):
        right = convert_to_messages(right)

    try:
        merged = left.copy()
        for message in right:
            if not message.id:
                message.id = str(uuid4())

            for index, existing in enumerate(merged):
                if existing.id == message.id:
                    merged[index] = message
                    break
            else:
                merged.append(message)
    except Exception as exc:
        logger.error("Error reducing messages: %s", exc)
        return left

    return merged


def reduce_todo_list(left: dict[str, bool], right: dict[str, bool]) -> dict[str, bool]:
    """Merge TODO lists by item name."""
    merged = left.copy()
    merged.update(right)
    return merged


def reduce_todo_lists(
    left: dict[str, dict[str, bool]],
    right: dict[str, dict[str, bool]],
) -> dict[str, dict[str, bool]]:
    """Merge named TODO lists; completed items win over incomplete ones."""
    merged = {name: items.copy() for name, items in left.items()}
    for name, items in right.items():
        if name in merged:
            for item, done in items.items():
                merged[name][item] = merged[name].get(item, False) or done
        else:
            merged[name] = items.copy()
    return merged


def reduce_current_agent_args(left: dict, right: dict) -> dict:
    return right


def reduce_current_tool_call(left, right):
    return right


def reduce_current_agent_report(left: str, right: str) -> str:
    return right


def reduce_is_finished(left: bool, right: bool) -> bool:
    return left or right


def reduce_is_cancelled(left: bool, right: bool) -> bool:
    return left or right


def reduce_progress(
    left: dict[str, Progress],
    right: dict[str, Progress],
) -> dict[str, Progress]:
    """Keep the entry with higher scheduled or finished counts."""
    merged = left.copy()
    for agent_name, progress in right.items():
        if agent_name not in merged:
            merged[agent_name] = progress
            continue

        existing = merged[agent_name]
        if (
            progress.scheduled_executions > existing.scheduled_executions
            or progress.finished_executions > existing.finished_executions
        ):
            merged[agent_name] = progress
    return merged


def reduce_iteration_number(left: int, right: int) -> int:
    return right


def reduce_file_refs(left: list, right: list) -> list:
    """Merge file reference lists by item id."""
    merged: dict = {}
    for ref in left:
        ref_id = getattr(ref, "id", None) or ref.get("id")
        if ref_id is not None:
            merged[ref_id] = ref
    for ref in right:
        ref_id = getattr(ref, "id", None) or ref.get("id")
        if ref_id is not None:
            merged[ref_id] = ref
    return list(merged.values())


def reducer_upsert(existing: list, new: list) -> list:
    """Upsert list items by ``id`` attribute; last write wins."""
    if not new:
        return existing

    new_by_id: dict = {}
    for item in new:
        item_id = getattr(item, "id", None)
        if item_id is None:
            raise ValueError(
                "reducer_upsert requires all items to have an 'id' attribute, "
                f"got {type(item).__name__} without one."
            )
        new_by_id[item_id] = item

    result = []
    seen_ids: set = set()
    for item in existing:
        item_id = getattr(item, "id", None)
        if item_id is None:
            raise ValueError(
                "reducer_upsert requires all items to have an 'id' attribute, "
                f"got {type(item).__name__} without one."
            )
        if item_id in new_by_id:
            result.append(new_by_id[item_id])
            seen_ids.add(item_id)
        else:
            result.append(item)
            seen_ids.add(item_id)

    for item in new:
        if item.id not in seen_ids:
            result.append(new_by_id[item.id])
            seen_ids.add(item.id)

    return result


def reduce_subagent_stack(left: list, right: list) -> list:
    return right if right is not None else left
