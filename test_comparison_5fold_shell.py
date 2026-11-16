"""Static safety checks for the five-fold comparison shell script.

The script is never launched by this test: it starts patient training on a
GPU, so it must only be run manually on the server.  These checks read the
source and assert the contracts that make it safe to run alongside an
existing training queue.

The same reasoning applies to the two legacy parity scripts, which are
covered by ``test_parity_shell_safety``; this file only adds the five-fold
comparison script to that idea.
"""
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / 'scripts' / 'run_5fold_comparison.sh'


class FiveFoldShellSafetyTest(unittest.TestCase):
    def setUp(self):
        self.source = SCRIPT.read_bytes().decode('utf-8').replace('\r\n', '\n')

    def test_strict_shell_mode_and_deterministic_python_hash_seed(self):
        self.assertIn('set -euo pipefail', self.source)
        self.assertIn('set -o noclobber', self.source)
        self.assertIn('export PYTHONHASHSEED=0', self.source)
        # The script must never delete anything.
        self.assertNotRegex(self.source, r'(?m)^\s*(?:sudo\s+)?rm\b')

    def test_new_batch_is_exclusive_and_inputs_are_overridable(self):
        self.assertIn('${COMPARISON_INPUTS_ROOT:-', self.source)
        self.assertIn('${COMPARISON_BATCH_OUT:-', self.source)
        self.assertIn('${COMPARISON_DEVICE:-', self.source)
        self.assertIn('${COMPARISON_EPOCHS:-', self.source)
        self.assertIn('[[ -e "$BATCH_OUT" || -L "$BATCH_OUT" ]]', self.source)
        self.assertIn('mkdir -- "$BATCH_OUT"', self.source)
        self.assertNotIn('mkdir -p -- "$BATCH_OUT"', self.source)

    def test_each_run_checks_plan_output_log_and_command_before_writing(self):
        for variable in ('plan', 'output', 'log', 'command_file'):
            self.assertIn(f'-e "${variable}"', self.source)
            self.assertIn(f'-L "${variable}"', self.source)
        self.assertIn("printf '%q ' \"${plan_command[@]}\" > \"$command_file\"",
                      self.source)
        self.assertNotRegex(self.source, r'>\|')

    def test_the_outer_test_partition_is_never_requested(self):
        # The comparison protocol rejects any manifest row that is not
        # train/val, so the script must not try to open a test partition.
        self.assertNotIn('outer_test', self.source.replace(
            'outer_test=not_loaded', ''))
        self.assertIn('outer_test=not_loaded', self.source)

    def test_the_dct_reference_is_loaded_and_never_rerun(self):
        self.assertIn('--dct-reference reference/dct_5fold_archive.json',
                      self.source)
        self.assertNotIn('comparison_5fold.py run --plan "$plan" '
                         '--plan-sha256 "$plan_sha" --model dct_reference',
                         self.source)

    def test_the_summary_refuses_an_existing_file(self):
        self.assertIn('SUMMARY="$BATCH_OUT/summary.json"', self.source)
        self.assertIn('if [[ -e "$SUMMARY" ]]; then', self.source)

    def test_every_arm_of_the_comparison_is_in_the_default_model_list(self):
        self.assertRegex(
            self.source,
            r'COMPARISON_MODELS:-pibd_2m camr_2m deref_2m dimaf_2m '
            r'rna_ridge wsi_only')
        self.assertIn('COMPARISON_FOLDS:-0 1 2 3 4', self.source)

    def test_the_script_never_starts_itself_or_loops_in_the_background(self):
        # A background launch without a recorded PID would be unmonitorable.
        launched = re.findall(r'(?m)^\s*\S.*&\s*$', self.source)
        self.assertEqual(launched, [])


if __name__ == '__main__':
    unittest.main()
