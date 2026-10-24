"""Independent audit of offline atomic-publication boundaries.

All roots, gates and lineage files are synthetic and temporary. No native
environment, historical dataset, production journal or service is accessed.
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from eventwm_graph.config import read_json, write_json, file_digest
from eventwm_graph.partial.dataset import iter_roots
from eventwm_graph.partial.report import summarize
from eventwm_graph.partial.stage1_acceptance import accept
from eventwm_graph.partial.storage import seal_root
from eventwm_graph.stage1.storage import scene_split
from test_stage1_review import fixture


class IndependentOfflinePublicationReview(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(); self.addCleanup(self.directory.cleanup)
        self.dataset = Path(self.directory.name)/'dataset'
        record,tape,queries,outcomes = fixture()
        record['scene_id'] = next('independent_offline_publication_'+str(i) for i in range(100)
                                  if scene_split('independent_offline_publication_'+str(i))=='train')
        record['sampling']['query_diagnostics'] = {
            'spatial_candidates':[],'rejected':[],'accepted_by_category':{'goal:':1}}
        self.root = seal_root(self.dataset,record,tape,queries,outcomes,min_free_bytes=0)
        self.folder = Path(self.root['folder'])
        self.stage0 = Path(self.directory.name)/'stage0.json'; write_json(self.stage0,{'status':'PASS'})
        self.replays = Path(self.directory.name)/'replays.json'; write_json(self.replays,{'status':'PASS','roots':[]})
        self.lineage = Path(self.directory.name)/'lineage.json'
        write_json(self.lineage,{'status':'PASS','roots':[{
            'root_id':self.root['root_id'],'passed':True,'full_queries_reproduced':1,
            'manifest_sha256':file_digest(self.folder/'manifest.json')}]})
        self.review = Path(self.directory.name)/'review.md'; self.review.write_text('Synthetic offline review only.\n')

    def staging(self):
        torn = self.folder.parent/'.partial_torn'; torn.mkdir()
        (torn/'manifest.json').write_text('{ torn unpublished manifest')
        sealed = self.folder.parent/'.partial_sealed_before_rename'; sealed.mkdir()
        # A complete-looking sealed flag can exist before atomic publication.
        # A held-out-looking identity must not even be opened by the scan.
        write_json(sealed/'manifest.json',{**read_json(self.folder/'manifest.json'),'split':'confirmation'})
        return {torn/'manifest.json',sealed/'manifest.json'}

    def accept(self):
        with patch('eventwm_graph.partial.stage1_acceptance.verify_gate'), \
                patch('eventwm_graph.partial.stage1_acceptance.run_unit_review',return_value={'passed':True}):
            return accept(self.stage0,[self.dataset],[self.replays],
                          Path(self.directory.name)/'temporary_acceptance',self.review,self.lineage)

    def test_acceptance_skips_torn_and_sealed_staging_before_any_manifest_open(self):
        staged = self.staging()
        def published_only(path):
            self.assertNotIn(Path(path),staged,'Unpublished staging manifest was opened')
            return read_json(path)
        with patch('eventwm_graph.partial.stage1_acceptance.read_json',side_effect=published_only):
            result = self.accept()
        self.assertEqual([r['root_id'] for r in result['roots']],[self.root['root_id']])
        self.assertEqual(result['evidence']['outcomes'],1)
        # One toy root is insufficient to open the scientific integration gate.
        self.assertEqual(result['status'],'FAIL')

    def test_reader_and_report_count_only_published_train_root(self):
        self.staging()
        self.assertEqual(list(iter_roots(self.dataset,'train')),[self.folder])
        result = summarize([self.dataset],Path(self.directory.name)/'temporary_report.json')
        self.assertEqual([r['root_id'] for r in result['roots']],[self.root['root_id']])
        self.assertEqual(result['outcomes'],1)

    def test_malformed_published_manifest_still_fails_closed_in_every_scanner(self):
        published = self.folder.parent/'published_corrupt_root'; published.mkdir()
        (published/'manifest.json').write_text('{ corrupt published manifest')
        with self.assertRaises(json.JSONDecodeError):list(iter_roots(self.dataset,'train'))
        with self.assertRaises(json.JSONDecodeError):
            summarize([self.dataset],Path(self.directory.name)/'temporary_report.json')
        with self.assertRaises(json.JSONDecodeError):self.accept()


if __name__=='__main__':unittest.main()
