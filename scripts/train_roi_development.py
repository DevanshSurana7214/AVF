"""Release training driver for authorized, locally prepared ROI data.

Implements the development protocol with validation-reused checkpoint selection.
This is newly assembled release orchestration, not the script that generated the
archived manuscript numbers. It never accesses an original outer test partition.
"""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from avf import MaskNet, drop_one, make_weights, preprocess

ARMS = {"none_no_drop":("none",False),"class_no_drop":("class",False),"joint_no_drop":("joint",False),
        "none_drop":("none",True),"class_drop":("class",True),"joint_drop":("joint",True),"logitadj_drop":("logitadj",True)}
MODS = ["mri","fdg","amy","tau"]


def predict(model,x,device):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(x[i:i+256].to(device)).softmax(1).cpu() for i in range(0,len(x),256)]).numpy()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    parser.add_argument("--arms",nargs="+",choices=list(ARMS),default=list(ARMS))
    args=parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.data_dir.resolve() == args.output_dir.resolve():
        raise ValueError("Input and output directories must differ")
    # Refuse output locations outside the ignored local_outputs tree in this checkout.
    safe=Path(__file__).resolve().parents[1]/"local_outputs"
    if not args.output_dir.resolve().is_relative_to(safe.resolve()):
        raise ValueError("Keep participant-level outputs under this checkout's ignored local_outputs directory")
    args.output_dir.mkdir(parents=True,exist_ok=False)
    ep=pd.read_csv(args.data_dir/"episodes.csv").set_index("episode_id").sort_index()
    if not ep.index.is_unique or ep[["participant_id","y"]].isna().any().any():
        raise ValueError("Unique episodes and nonmissing participant groups/labels required")
    if set(ep.y.unique()) != {0,1,2}:
        raise ValueError("The three diagnoses must be encoded 0,1,2")
    mask=ep[[f"m_{m}" for m in MODS]].to_numpy()
    if not np.isin(mask,[0,1]).all() or (mask.sum(1)==0).any():
        raise ValueError("Nonempty binary availability masks required")
    blocks={m:pd.read_csv(args.data_dir/f"{m}.csv").set_index("episode_id").reindex(ep.index) for m in MODS}
    if any(not block.index.is_unique for block in blocks.values()):
        raise ValueError("Unique feature-row episode keys required")
    y=ep.y.to_numpy(dtype=int); patterns=mask @ np.array([8,4,2,1])
    splitter=StratifiedGroupKFold(3,shuffle=True,random_state=624)
    (args.output_dir/"protocol.json").write_text(json.dumps({"development_only":True,"checkpoint_validation_reused":True,
        "original_outer_validation_or_test":False,"seeds":[42,1042,2042],"arms":args.arms,"device":args.device,
        "not_a_claim_to_reproduce_archived_scores":True},indent=2))
    for fold,(train,val) in enumerate(splitter.split(ep,y,ep.participant_id)):
        if set(ep.iloc[train].participant_id)&set(ep.iloc[val].participant_id):
            raise AssertionError("Participant overlap")
        folder=args.output_dir/f"fold{fold}"; folder.mkdir()
        x,state,slices=preprocess(ep,blocks,ep.index[train],{"max_missing_feature_fraction":.2,"z_clip":5})
        (folder/"preprocessing.json").write_text(json.dumps({"state":state,"slices":slices},indent=2))
        cw,jw,jw_nat,prior,weights=make_weights(y[train],patterns[train])
        (folder/"weights.json").write_text(json.dumps(weights,indent=2))
        tensor=torch.tensor(x)
        for seed in [42,1042,2042]:
            params=dict(objective="multiclass",num_class=3,metric="multi_logloss",num_leaves=31,learning_rate=.03,
                        min_data_in_leaf=20,feature_fraction=.8,bagging_fraction=.8,bagging_freq=1,num_threads=3,
                        verbosity=-1,seed=seed,deterministic=True,force_col_wise=True)
            tree=lgb.train(params,lgb.Dataset(x[train],label=y[train]),num_boost_round=800,
                valid_sets=[lgb.Dataset(x[val],label=y[val])],callbacks=[lgb.early_stopping(60,verbose=False)])
            tree.save_model(str(folder/f"seed{seed}_tree.txt"))
            predictions={"mask_lightgbm":tree.predict(x[val])}
            for arm in args.arms:
                scheme,use_drop=ARMS[arm]
                random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                if args.device=="cuda":torch.cuda.manual_seed_all(seed)
                net=MaskNet(slices).to(args.device)
                opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
                classw=torch.tensor(cw,device=args.device)
                jointw=torch.tensor(jw if use_drop else jw_nat,device=args.device)
                priorlog=torch.tensor(np.log(prior),device=args.device)
                best=-1.;stale=0; checkpoint=folder/f"seed{seed}_{arm}_best.pt"
                for epoch in range(30):
                    net.train();order=np.random.default_rng(seed+epoch).permutation(train)
                    for start in range(0,len(order),128):
                        ids=order[start:start+128];batch=tensor[ids].to(args.device)
                        labels=torch.tensor(y[ids],device=args.device)
                        if use_drop:batch=drop_one(batch,slices)
                        effective=(batch[:,-4:].long()*torch.tensor([8,4,2,1],device=args.device)).sum(1)
                        w=torch.ones(len(ids),device=args.device)
                        if scheme=="class":w=classw[labels]
                        elif scheme=="joint":w=jointw[effective,labels]
                        logits=net(batch)
                        if scheme=="logitadj":logits=logits+priorlog[effective]
                        loss=(torch.nn.functional.cross_entropy(logits,labels,reduction="none")*w).mean()
                        if not torch.isfinite(loss):raise RuntimeError("Nonfinite loss")
                        opt.zero_grad();loss.backward()
                        torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
                    p=predict(net,tensor[val],args.device)
                    score=np.mean([roc_auc_score(y[val]==c,p[:,c]) for c in range(3)])
                    if score>best:
                        best=float(score);stale=0
                        torch.save({"model":net.state_dict(),"epoch":epoch+1},checkpoint)
                    else:stale+=1
                    if stale>=8:break
                net.load_state_dict(torch.load(checkpoint,map_location=args.device,weights_only=True)["model"])
                predictions[arm]=predict(net,tensor[val],args.device)
                del net,opt
                if args.device=="cuda":torch.cuda.empty_cache()
            rows=[]
            for arm,p in predictions.items():
                frame=ep.iloc[val][["participant_id","y"]].copy()
                frame["pattern"]=patterns[val];frame["fold"]=fold;frame["seed"]=seed;frame["method"]=arm
                frame[["p0","p1","p2"]]=p; rows.append(frame.reset_index())
            pd.concat(rows).to_csv(folder/f"seed{seed}_private_predictions.csv",index=False)
            print(json.dumps({"fold":fold,"seed":seed,"status":"saved locally","original_test_inference":False}),flush=True)


if __name__=="__main__":main()
