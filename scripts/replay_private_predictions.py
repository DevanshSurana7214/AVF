"""Replay the frozen AVF mixture using authorized local OOF prediction files.

No identifiers are printed or uploaded. This is prediction replay, not retraining
or proof of independent generalization. Fixed alphas apply to the archived folds.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from avf import fuse, evaluate, supported_cells


def load(path):
    d=pd.read_csv(path)
    d=d.rename(columns={"RID":"participant_id","effective_pattern":"pattern"})
    if "condition" in d:
        d=d[d.condition=="natural"].copy()
    required=["episode_id","participant_id","y","pattern","fold","seed","p0","p1","p2"]
    if not set(required)<=set(d) or d[required].isna().any().any():
        raise ValueError("Missing required private prediction columns/values")
    if d.duplicated(["episode_id","seed"]).any():
        raise ValueError("Supply exactly one model's aligned predictions per file")
    if not np.isfinite(d[["p0","p1","p2"]]).all().all():
        raise ValueError("Nonfinite prediction")
    if (d.groupby("episode_id")[["participant_id","y","pattern","fold"]].nunique()!=1).any().any():
        raise ValueError("Episode metadata differs between seeds")
    if (d.groupby("episode_id").seed.nunique()!=3).any() or set(d.seed)!={42,1042,2042}:
        raise ValueError("Three expected historical seeds required")
    if (d.groupby("participant_id").fold.nunique()!=1).any():
        raise ValueError("Participant occurs in more than one held-out fold")
    return d.groupby("episode_id",as_index=False).agg({"participant_id":"first","y":"first","pattern":"first","fold":"first","p0":"mean","p1":"mean","p2":"mean"}).sort_values("episode_id")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree-csv",type=Path,required=True)
    parser.add_argument("--neural-csv",type=Path,required=True)
    parser.add_argument("--frozen-folds-confirmed",action="store_true",help="Confirm these are the historical cohort/fold identities, not a new dataset")
    args=parser.parse_args()
    if not args.frozen_folds_confirmed:
        raise ValueError("Frozen fold identities must be verified locally before using archived alphas")
    a,b=load(args.tree_csv),load(args.neural_csv)
    keys=["episode_id","participant_id","y","pattern","fold"]
    if not a[keys].reset_index(drop=True).equals(b[keys].reset_index(drop=True)):
        raise ValueError("Tree and neural predictions do not align")
    if len(a)!=9566 or a.participant_id.nunique()!=2260 or set(a.fold)!={0,1,2}:
        raise ValueError("This is not the frozen paper's development cohort")
    cells=supported_cells(a.y,a.pattern,a.participant_id)
    if len(cells)!=34:raise ValueError("Frozen SCR support mismatch")
    pa=a[["p0","p1","p2"]].to_numpy();pb=b[["p0","p1","p2"]].to_numpy()
    pf=np.empty_like(pa)
    for fold,alpha in {0:.55,1:.60,2:.55}.items():
        use=a.fold.to_numpy()==fold;pf[use]=fuse(pa[use],pb[use],alpha)
    print(json.dumps({"scope":"frozen development prediction replay; not nested or external confirmation",
                      "episodes":len(a),"participants":a.participant_id.nunique(),"supported_cells":len(cells),
                      "tree":evaluate(a.y,pa,a.pattern,cells),"neural_joint_drop":evaluate(a.y,pb,a.pattern,cells),
                      "avf":evaluate(a.y,pf,a.pattern,cells)},indent=2))


if __name__=="__main__":main()
