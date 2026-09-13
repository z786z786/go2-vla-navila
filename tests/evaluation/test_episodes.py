import ast
import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from evaluation.episodes import (DEFAULT_CONFIG, DEFAULT_SOURCE, SOURCE_SHA256,
                                 load_dev100, load_episodes)


class EpisodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = json.loads(gzip.decompress(DEFAULT_SOURCE.read_bytes()))
        cls.loaded = load_episodes()

    def fixture(self, data):
        directory = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / 'source.json.gz'
        raw = gzip.compress(json.dumps(data).encode())
        path.write_bytes(raw)
        return path, hashlib.sha256(raw).hexdigest()

    def test_source_checksum(self):
        self.assertEqual(hashlib.sha256(DEFAULT_SOURCE.read_bytes()).hexdigest(), SOURCE_SHA256)
        self.assertEqual(len(self.loaded), 1077)

    def test_checksum_rejected_before_decompression(self):
        path, checksum = self.fixture(self.raw)
        path.write_bytes(path.read_bytes() + b'corruption')
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch: expected .* got .*'):
            load_episodes(path, expected_sha256=checksum)

    def test_instruction_literal_parsing(self):
        data = copy.deepcopy(self.raw)
        expected = {'instruction_text': "Turn left; don't stop", 'instruction_tokens': [1, 2]}
        data['episodes'][0]['instruction'] = repr(expected)
        path, checksum = self.fixture(data)
        self.assertEqual(load_episodes(path, expected_sha256=checksum)[0]['instruction'], expected)

    def test_reference_path_literal_parsing(self):
        self.check_path_literal('reference_path')

    def test_gt_locations_literal_parsing(self):
        self.check_path_literal('gt_locations')

    def check_path_literal(self, field):
        data = copy.deepcopy(self.raw)
        expected = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        data['episodes'][0][field] = repr(expected)
        path, checksum = self.fixture(data)
        self.assertEqual(load_episodes(path, expected_sha256=checksum)[0][field], expected)

    def test_literal_parser_rejects_executable_expression(self):
        data = copy.deepcopy(self.raw)
        data['episodes'][0]['instruction'] = "dict(instruction_text='unsafe expression')"
        path, checksum = self.fixture(data)
        with self.assertRaises(ValueError):
            load_episodes(path, expected_sha256=checksum)

    def test_all_radii(self):
        self.assertEqual(len(self.loaded), 1077)
        for episode in self.loaded:
            self.assertEqual(episode['goals'][0]['radius'], 3.0)

    def test_radius_error_lists_every_offender(self):
        data = copy.deepcopy(self.raw)
        for episode in data['episodes'][:2]:
            episode['goals'][0]['radius'] = 2.0
        path, checksum = self.fixture(data)
        with self.assertRaises(AssertionError) as caught:
            load_episodes(path, expected_sha256=checksum)
        for episode in data['episodes'][:2]:
            self.assertIn(str((episode['episode_id'], 2.0)), str(caught.exception))

    def test_wrong_population_rejected(self):
        data = {'episodes': self.raw['episodes'][:-1]}
        path, checksum = self.fixture(data)
        with self.assertRaisesRegex(AssertionError, 'Expected 1077 episodes, got 1076'):
            load_episodes(path, expected_sha256=checksum)

    def test_dev100_manifest_order_and_real_paths(self):
        manifest = json.loads(DEFAULT_CONFIG.read_text())
        subset = load_dev100()
        self.assertEqual([e['episode_id'] for e in subset], manifest['episode_ids'])
        self.assertEqual(len(subset), 100)
        originals = {e['episode_id']: e for e in self.loaded}
        for episode in subset:
            self.assertEqual(episode['gt_locations'], originals[episode['episode_id']]['gt_locations'])

    def test_dev100_is_derived_from_supplied_config(self):
        directory = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(directory.cleanup)
        config = Path(directory.name) / 'dev.json'
        manifest = json.loads(DEFAULT_CONFIG.read_text())
        manifest['episode_ids'].reverse()
        config.write_text(json.dumps(manifest))
        self.assertEqual([e['episode_id'] for e in load_dev100(config=config)], manifest['episode_ids'])

    def test_decoded_fields_remain_equal(self):
        for field in ('instruction', 'reference_path', 'gt_locations'):
            original = self.raw['episodes'][0][field]
            expected = ast.literal_eval(original) if isinstance(original, str) else original
            self.assertEqual(self.loaded[0][field], expected)
