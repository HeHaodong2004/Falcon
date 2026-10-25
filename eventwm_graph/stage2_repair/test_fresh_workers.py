import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from torch.utils.data import DataLoader
from .fresh_data import FreshGroups,keep


class PersistentSampling(unittest.TestCase):
    def test_epoch_changes_reach_persistent_workers(self):
        with tempfile.TemporaryDirectory()as temp:
            p=Path(temp);n=1000;x=np.zeros((4*n,88),np.float32);x[:,18:20]=1.;y=np.zeros(4*n,np.uint8);keys=np.column_stack([np.arange(4*n)-2000,-np.arange(4*n)])
            for i in range(4):x[i*n:i*n+500,18]=.01;y[i*n:i*n+5]=1
            np.savez(p/'points.npz',x=x,y=y,extra=np.zeros((4*n,16),np.float32),keys=keys)
            maps=np.zeros((4,14,8,8),np.float16)
            for i in range(4):maps[i,0]=i
            np.savez(p/'maps.npz',maps=maps,lo=np.zeros((4,2),int),aux=np.zeros((4,8,8),np.float16),mask=np.zeros((4,8,8),bool))
            row=dict(root_id='training_fixture',scene_id='fixture_scene',point_path=str(p/'points.npz'),map_path=str(p/'maps.npz'),groups=[dict(map_index=i,segments=[dict(start=i*n,end=(i+1)*n)])for i in range(4)])
            (p/'manifest.json').write_text(json.dumps(dict(roots=1,scenes=1,records=[row])))
            data=FreshGroups(p/'manifest.json');loader=DataLoader(data,batch_size=2,num_workers=2,collate_fn=keep,persistent_workers=True,multiprocessing_context='fork');results=[]
            for epoch in [1,2]:
                data.epoch.value=epoch;out={}
                for batch in loader:
                    for g in batch:out[int(g['maps'][0,0,0])]=g['indices'].copy();self.assertAlmostEqual(float(g['weight'].sum()),.25,places=7)
                results.append(out)
            self.assertEqual(set(results[0]),set(range(4)));self.assertEqual(set(results[1]),set(range(4)))
            for i in range(4):self.assertFalse(np.array_equal(results[0][i],results[1][i]))
            del loader


if __name__=='__main__':unittest.main()
