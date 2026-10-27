import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import torch
from avf import MaskNet, drop_one, make_weights, preprocess, fuse, evaluate, supported_cells
from avf.executed_components import adjusted, pbar_mat

torch.set_num_threads(2)


class TestAVF(unittest.TestCase):
    def test_expected_dropout_mass_and_normalization(self):
        y=np.zeros(100,dtype=int); pattern=np.full(100,12)
        cw,jw,jw_nat,pi,report=make_weights(y,pattern)
        e=np.asarray(report["expected_dropout_cell_counts"])
        self.assertEqual(e[12,0],50);self.assertEqual(e[8,0],25);self.assertEqual(e[4,0],25)
        self.assertEqual(e.sum(),100)
        self.assertAlmostEqual(float((e*jw).sum()/100),1,places=6)
        np.testing.assert_allclose(pi.sum(1),1,atol=1e-6)

    def test_every_pattern_preserves_mass(self):
        pattern=np.repeat(np.arange(1,16),3);y=np.tile(np.arange(3),15)
        *_,report=make_weights(y,pattern)
        e=np.asarray(report["expected_dropout_cell_counts"])
        self.assertAlmostEqual(float(e.sum()),45,places=12)
        self.assertEqual(e[0].sum(),0)
        self.assertTrue(np.isfinite(np.asarray(report["joint_weights_dropout_basis"])).all())

    def test_masked_branches_do_not_affect_output(self):
        torch.manual_seed(3)
        slices={m:(i*2,(i+1)*2) for i,m in enumerate(["mri","fdg","amy","tau"])}
        model=MaskNet(slices).eval()
        x=torch.randn(8,12);x[:,-4:]=torch.tensor([1,0,1,0])
        changed=x.clone();changed[:,2:4]=10000;changed[:,6:8]=-10000
        with torch.no_grad():
            np.testing.assert_allclose(model(x).numpy(),model(changed).numpy(),atol=1e-6)

    def test_dropout_removes_one_and_keeps_singletons(self):
        slices={m:(i,i+1) for i,m in enumerate(["mri","fdg","amy","tau"])}
        x=torch.ones(30,8);x[0,-4:]=torch.tensor([1,0,0,0]);x[0,1:4]=0
        torch.manual_seed(5);out=drop_one(x,slices,probability=1)
        self.assertEqual(out[0,-4:].sum(),1)
        self.assertTrue((out[1:,-4:].sum(1)==3).all())
        self.assertTrue((out[:,-4:].sum(1)>0).all())
        self.assertTrue(torch.equal(x[0],out[0]))

    def test_preprocessing_uses_training_only_statistics(self):
        episodes=pd.DataFrame(np.ones((3,4)),columns=["m_mri","m_fdg","m_amy","m_tau"])
        blocks={m:pd.DataFrame({"feature":[2.,4.,999.]}) for m in ["mri","fdg","amy","tau"]}
        _,state,_=preprocess(episodes,blocks,[0,1],{"max_missing_feature_fraction":.2,"z_clip":5})
        self.assertEqual(state["mri"]["mean"],[3.]);self.assertEqual(state["mri"]["sd"],[1.])
        blocks["mri"].loc[2,"feature"]=-999999.
        _,second,_=preprocess(episodes,blocks,[0,1],{"max_missing_feature_fraction":.2,"z_clip":5})
        self.assertEqual(state,second)

    def test_fusion_and_zero_correction(self):
        a=np.array([[.2,.5,.3],[.7,.2,.1]]);b=np.array([[.3,.6,.1],[.4,.2,.4]])
        p=fuse(a,b,.55);np.testing.assert_allclose(p.sum(1),1)
        prior=pbar_mat(np.zeros((16,3)),[5,3,2])
        np.testing.assert_array_equal(adjusted(p,np.array([8,12]),prior,0),p)
        corrected=adjusted(p,np.array([8,12]),prior,1)
        np.testing.assert_allclose(corrected.sum(1),1)
        self.assertTrue(np.isfinite(corrected).all())

    def test_scr_support_counts_subjects_not_visits(self):
        y=np.array([0,0,0,1,1,2,2]);pat=np.full(7,8);subjects=["synthetic-A"]*3+["synthetic-B","synthetic-C","synthetic-D","synthetic-E"]
        cells=supported_cells(y,pat,subjects,min_subjects=2)
        self.assertNotIn((8,0),cells);self.assertIn((8,1),cells);self.assertIn((8,2),cells)
        p=np.eye(3)[y]
        m=evaluate(y,p,pat,cells)
        self.assertEqual(m["supported_cell_recall"],1)
        self.assertEqual(m["macro_auroc"],1)

    def test_frozen_public_table_matches_paper(self):
        root=Path(__file__).resolve().parents[1]
        data=pd.read_csv(root/"results/natural_results.csv")
        row=data.loc[data.method=="avf"].iloc[0]
        self.assertAlmostEqual(row.macro_auroc,.79692716877882)
        self.assertAlmostEqual(row.macro_f1,.6194397287578783)
        self.assertAlmostEqual(row.supported_cell_recall,.6042585170011627)


if __name__=="__main__":unittest.main()
