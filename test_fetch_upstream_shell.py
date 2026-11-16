"""Static safety checks for the upstream comparator fetch script.

The script clones four third-party repositories, so it must only be run
manually on the server.  These checks read the source and assert the
contracts that make it safe: it never deletes or resets anything, it refuses
to overwrite a dirty checkout, and it pins every commit.

The script is never launched by this test.
"""
from pathlib import Path
import re
import unittest

import comparison_sources as sources


ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / 'scripts' / 'fetch_upstream_comparators.sh'


class FetchUpstreamShellSafetyTest(unittest.TestCase):
    def setUp(self):
        self.source = SCRIPT.read_bytes().decode('utf-8').replace('\r\n', '\n')

    def test_strict_shell_mode(self):
        self.assertIn('set -euo pipefail', self.source)

    def test_the_script_never_deletes_or_resets_anything(self):
        # A reset would discard work inside a third-party checkout, and a
        # delete would destroy it.  Neither may appear.
        self.assertNotRegex(self.source, r'(?m)^\s*(?:sudo\s+)?rm\b')
        self.assertNotIn('reset --hard', self.source)
        self.assertNotIn('git reset', self.source)
        self.assertNotIn('checkout -- .', self.source)
        self.assertNotIn('--force', self.source)

    def test_a_dirty_checkout_stops_the_script(self):
        self.assertIn('git -C "$target" status --porcelain', self.source)
        self.assertIn('refusing to touch it', self.source)

    def test_every_repository_is_pinned_to_the_recorded_commit(self):
        for name, record in sources.UPSTREAM.items():
            self.assertIn(record['commit'], self.source, name)
        self.assertIn('does not match the pinned commit', self.source)

    def test_the_script_uses_the_recorded_urls(self):
        for name, record in sources.UPSTREAM.items():
            self.assertIn(record['url'], self.source, name)

    def test_the_upstream_root_matches_the_registry(self):
        # The script must resolve the same directory that comparison_sources.py
        # uses, otherwise the hash verification would read a different tree.
        self.assertIn('COMPARATOR_ROOT:-$(cd -- "$CODE_DIR/.." && pwd)/papers/comparators',
                      self.source)
        self.assertEqual(sources.COMPARATORS.name, 'comparators')
        self.assertEqual(sources.COMPARATORS.parent.name, 'papers')

    def test_the_script_never_writes_into_an_upstream_tree(self):
        # Only git clone/checkout/fetch may touch the checkout directory.
        writes = re.findall(r'(?m)^(?!\s*#)(.*>\s*"?\$COMPARATORS.*)$', self.source)
        self.assertEqual(writes, [])

    def test_the_script_verifies_the_recorded_hashes_at_the_end(self):
        self.assertIn('sources.verify_upstream()', self.source)
        self.assertIn('all upstream checkouts match the recorded hashes', self.source)

    def test_the_script_never_starts_a_background_job(self):
        launched = re.findall(r'(?m)^\s*\S.*&\s*$', self.source)
        self.assertEqual(launched, [])


if __name__ == '__main__':
    unittest.main()
