"""Testes de extração do último par user/assistant para RAG."""

from __future__ import annotations

import unittest

from tcc.rag.index import example_to_text, last_assistant_message, last_user_message


class LastTurnRagExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.item = {
            "messages": [
                {"role": "system", "content": "You are a planner."},
                {"role": "user", "content": "Here are two examples. Demo Q1"},
                {"role": "assistant", "content": "Node:\n1: demo1"},
                {"role": "user", "content": "Demo Q2"},
                {"role": "assistant", "content": "Node:\n1: demo2"},
                {"role": "user", "content": "Now it's your turn. Real question"},
                {"role": "assistant", "content": "Node:\n1: real gold"},
            ]
        }

    def test_last_user_is_real_task(self) -> None:
        self.assertEqual(
            last_user_message(self.item),
            "Now it's your turn. Real question",
        )

    def test_last_assistant_is_gold(self) -> None:
        self.assertEqual(last_assistant_message(self.item), "Node:\n1: real gold")

    def test_example_to_text_query_only(self) -> None:
        self.assertEqual(
            example_to_text(self.item, ["query"]),
            "Now it's your turn. Real question",
        )

    def test_example_to_text_workflow_only(self) -> None:
        self.assertEqual(example_to_text(self.item, ["workflow"]), "Node:\n1: real gold")

    def test_ignores_system_and_demos(self) -> None:
        q = last_user_message(self.item)
        w = last_assistant_message(self.item)
        self.assertNotIn("Here are two examples", q)
        self.assertNotIn("Demo Q2", q)
        self.assertNotIn("demo1", w)
        self.assertNotIn("demo2", w)
        self.assertNotIn("You are a planner", q)
        self.assertNotIn("You are a planner", w)


if __name__ == "__main__":
    unittest.main()
