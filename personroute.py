"""Prompt-based decision tree router.

This program reads a JSON decision tree and an order payload, then
returns the person the order should be routed to.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict


MISSING = object()


def _load_json_from_file(path: str) -> Any:
	with open(path, "r", encoding="utf-8") as handle:
		return json.load(handle)


def _load_json_from_text(text: str) -> Any:
	return json.loads(text)


def read_json_input(prompt: str) -> Any:
	while True:
		raw = input(prompt).strip()
		if not raw:
			print("Input cannot be empty.")
			continue

		if raw.startswith("{") or raw.startswith("["):
			try:
				return _load_json_from_text(raw)
			except json.JSONDecodeError as exc:
				print(f"Invalid JSON: {exc}")
				continue

		if os.path.exists(raw):
			try:
				return _load_json_from_file(raw)
			except (OSError, json.JSONDecodeError) as exc:
				print(f"Could not read JSON file: {exc}")
				continue

		print("Enter a valid JSON string or a file path.")


def _parse_value(raw: str) -> Any:
	raw = raw.strip()
	if not raw:
		return ""
	try:
		return json.loads(raw)
	except json.JSONDecodeError:
		return raw


def prompt_for_order() -> Dict[str, Any]:
	print("Enter order fields one by one. Leave the field name blank to finish.")
	data: Dict[str, Any] = {}
	while True:
		key = input("Field name: ").strip()
		if not key:
			break
		value = _parse_value(input("Value (JSON or plain text): "))
		data[key] = value
	return data


def _compare(operator: str, left: Any, right: Any) -> bool:
	if operator == "==":
		return left == right
	if operator == "!=":
		return left != right
	if operator == ">":
		return left > right
	if operator == ">=":
		return left >= right
	if operator == "<":
		return left < right
	if operator == "<=":
		return left <= right
	if operator == "in":
		return left in right
	if operator == "not_in":
		return left not in right
	if operator == "contains":
		return right in left
	if operator == "not_contains":
		return right not in left
	if operator == "exists":
		return left is not MISSING
	if operator == "not_exists":
		return left is MISSING
	raise ValueError(f"Unsupported operator: {operator}")


def evaluate_tree(node: Dict[str, Any], order: Dict[str, Any]) -> str:
	if "person" in node:
		return str(node["person"])

	attribute = node.get("attribute")
	operator = node.get("operator", "==")
	value = node.get("value")
	true_branch = node.get("true")
	false_branch = node.get("false")

	if attribute is None or true_branch is None or false_branch is None:
		raise ValueError("Decision node requires attribute, true, and false keys.")

	left = order.get(attribute, MISSING)
	result = _compare(operator, left, value)
	next_node = true_branch if result else false_branch
	if not isinstance(next_node, dict):
		raise ValueError("Branch nodes must be objects.")
	return evaluate_tree(next_node, order)


def _parse_cli_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Route an order to a person using a JSON decision tree."
	)
	parser.add_argument(
		"--tree",
		help="Path to JSON tree or raw JSON string.",
	)
	parser.add_argument(
		"--order",
		help="Path to order JSON or raw JSON string.",
	)
	return parser.parse_args()


def main() -> int:
	args = _parse_cli_args()

	if args.tree:
		tree = _load_json_from_file(args.tree) if os.path.exists(args.tree) else _load_json_from_text(args.tree)
	else:
		tree = read_json_input("Decision tree JSON (string or file path): ")

	if args.order:
		order = _load_json_from_file(args.order) if os.path.exists(args.order) else _load_json_from_text(args.order)
	else:
		choice = input("Provide order as JSON string/file? [y/N]: ").strip().lower()
		if choice == "y":
			order = read_json_input("Order JSON (string or file path): ")
		else:
			order = prompt_for_order()

	if not isinstance(tree, dict):
		print("Decision tree must be a JSON object.")
		return 1
	if not isinstance(order, dict):
		print("Order must be a JSON object.")
		return 1

	try:
		person = evaluate_tree(tree, order)
	except ValueError as exc:
		print(f"Error: {exc}")
		return 1

	print(f"Assigned person: {person}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
