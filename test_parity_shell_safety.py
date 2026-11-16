"""Static safety checks only: never launch either patient-training shell script."""
from pathlib import Path
import re
import shlex
import unittest


ROOT = Path(__file__).resolve().parent
SCRIPT_NAMES = ('run_aux_comparison.sh', 'run_parity_controls.sh')


class ParityShellSafetyTest(unittest.TestCase):
    def scripts(self):
        for name in SCRIPT_NAMES:
            data = (ROOT / 'scripts' / name).read_bytes()
            # Git may check out CRLF on Windows; these are source-contract
            # checks, not a claim that a Windows checkout was executed by bash.
            yield name, data.decode('utf-8').replace('\r\n', '\n')

    def test_strict_shell_mode_and_deterministic_python_hash_seed(self):
        for name, source in self.scripts():
            with self.subTest(script=name):
                self.assertIn('set -euo pipefail', source)
                self.assertIn('set -o noclobber', source)
                self.assertIn('export PYTHONHASHSEED=0', source)
                self.assertNotRegex(source, r'(?m)^\s*(?:sudo\s+)?rm\b')

    def test_new_batch_is_exclusive_and_inputs_are_overridable(self):
        for name, source in self.scripts():
            with self.subTest(script=name):
                self.assertIn('${PARITY_INPUTS_ROOT:-', source)
                self.assertIn('${PARITY_BATCH_OUT:-', source)
                self.assertIn('_corrected_nll_$(date -u', source)
                self.assertIn('[[ -e "$BATCH_OUT" || -L "$BATCH_OUT" ]]', source)
                self.assertIn('mkdir -- "$BATCH_OUT"', source)
                self.assertNotIn('mkdir -p -- "$BATCH_OUT"', source)

    def test_each_run_checks_directory_log_and_command_before_writing(self):
        for name, source in self.scripts():
            with self.subTest(script=name):
                for variable in ('output', 'log', 'command_file'):
                    self.assertIn(f'-e "${variable}"', source)
                    self.assertIn(f'-L "${variable}"', source)
                self.assertIn('"${command[@]}" > "$log" 2>&1', source)
                self.assertIn("printf '%q ' \"${command[@]}\" > \"$command_file\"", source)
                self.assertNotRegex(source, r'>\|')

    def test_every_background_job_has_an_explicit_checked_pid_wait(self):
        for name, source in self.scripts():
            with self.subTest(script=name):
                self.assertIn('if wait "${PIDS[$index]}"; then', source)
                self.assertIn('failed=1', source)
                self.assertIn('return "$failed"', source)
                self.assertNotRegex(source, r'(?m)^\s*wait\s*$')
                launched = re.findall(r'(?m)^run .+ &$', source)
                recorded = re.findall(r'(?m)^PIDS\+=\("\$!"\)$', source)
                self.assertEqual(len(launched), len(recorded))
                self.assertGreater(len(launched), 0)
                self.assertIn('if ! wait_for_jobs; then\n  exit 1\nfi', source)

    def test_both_auxiliary_arms_share_the_same_batch_and_tail_arguments(self):
        source = (ROOT / 'scripts' / SCRIPT_NAMES[0]).read_text(encoding='utf-8')
        body = re.search(r'local -a command=\(\n(.*?)\n  \)', source, flags=re.S).group(1)
        arguments = shlex.split(body)
        for option in ('--patient-batch-size', '--batch-tail-minimum'):
            self.assertEqual(arguments[arguments.index(option) + 1], '17')
        self.assertIn('--legacy-architecture', arguments)
        self.assertNotIn('--disable-auxiliary', arguments)
        self.assertIn('run aux_on "$DEVICE_0" &', source)
        self.assertIn('run aux_off "$DEVICE_1" --disable-auxiliary &', source)

    def test_controls_retain_legacy_auxoff_and_check_first_phase_before_next_job(self):
        source = (ROOT / 'scripts' / SCRIPT_NAMES[1]).read_text(encoding='utf-8')
        body = re.search(r'local -a command=\(\n(.*?)\n  \)', source, flags=re.S).group(1)
        arguments = shlex.split(body)
        self.assertIn('--legacy-architecture', arguments)
        self.assertIn('--disable-auxiliary', arguments)
        self.assertEqual(arguments[arguments.index('--patient-batch-size') + 1], '8')
        first_gate = source.index('if ! wait_for_jobs; then')
        next_job = source.index('run path_only ')
        self.assertLess(first_gate, next_job)
        self.assertNotIn('must\n#                 collapse to chance', source)
        self.assertNotIn('Everything else matches the reference run exactly', source)


if __name__ == '__main__':
    unittest.main()
