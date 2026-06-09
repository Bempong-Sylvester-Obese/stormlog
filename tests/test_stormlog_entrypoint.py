import contextlib
import io
import unittest
from unittest import mock

import stormlog.entrypoint as entrypoint


class StormlogEntrypointTests(unittest.TestCase):
    def test_no_args_launches_tui(self) -> None:
        with mock.patch("stormlog.tui.run_app") as run_app:
            exit_code = entrypoint.main([])

        self.assertEqual(exit_code, 0)
        run_app.assert_called_once_with()

    def test_tui_command_launches_tui(self) -> None:
        with mock.patch("stormlog.tui.run_app") as run_app:
            exit_code = entrypoint.main(["tui"])

        self.assertEqual(exit_code, 0)
        run_app.assert_called_once_with()

    def test_tui_help_does_not_launch_tui(self) -> None:
        output = io.StringIO()
        with mock.patch("stormlog.tui.run_app") as run_app:
            with contextlib.redirect_stdout(output):
                with self.assertRaises(SystemExit) as raised:
                    entrypoint.main(["tui", "--help"])

        self.assertEqual(raised.exception.code, 0)
        run_app.assert_not_called()
        self.assertIn("Stormlog Textual TUI", output.getvalue())

    def test_tui_unknown_arg_errors_before_launching_tui(self) -> None:
        with mock.patch("stormlog.tui.run_app") as run_app:
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    entrypoint.main(["tui", "--bad"])

        self.assertEqual(raised.exception.code, 2)
        run_app.assert_not_called()

    def test_infer_command_dispatches_to_infer_cli(self) -> None:
        with mock.patch("stormlog.infer.cli.main", return_value=7) as infer_main:
            exit_code = entrypoint.main(["infer", "analyze", "artifact.jsonl"])

        self.assertEqual(exit_code, 7)
        infer_main.assert_called_once_with(["analyze", "artifact.jsonl"])

    def test_help_mentions_no_arg_tui_behavior(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = entrypoint.main(["--help"])

        self.assertEqual(exit_code, 0)
        self.assertIn("Textual TUI", output.getvalue())

    def test_unknown_command_exits_with_parser_error(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                entrypoint.main(["unknown"])

        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
