"""Verbatim historical definitions extracted without importing private study loaders.
Do not edit this generated module manually. See docs/CODE_PROVENANCE.json.
The optional generator in historical drop_one does not control every random draw;
set torch global seeds as in the executed study. This behavior is retained.
"""
import numpy as np
import pandas as pd
import torch
from torch import nn
MODS=["mri","fdg","amy","tau"]
BITS=np.array([8,4,2,1])
KAPPA=20.


class MaskNet(nn.Module):
 def __init__(self,slices):
  super().__init__();self.slices=slices
  self.branches=nn.ModuleList([nn.Sequential(nn.Linear(b-a,64),nn.GELU(),nn.LayerNorm(64),nn.Dropout(.1),nn.Linear(64,64),nn.GELU()) for a,b in slices.values()])
  self.gate=nn.Linear(64,1);self.head=nn.Sequential(nn.Linear(68,64),nn.GELU(),nn.Dropout(.1),nn.Linear(64,3))
 def forward(self,x):
  mask=x[:,-4:];assert bool((mask.sum(1)>0).all())
  z=torch.stack([branch(x[:,a:b]) for branch,(a,b) in zip(self.branches,self.slices.values())],1)
  scores=self.gate(z).squeeze(-1).masked_fill(mask==0,-1e9);weight=scores.softmax(1);pooled=(z*weight[:,:,None]*mask[:,:,None]).sum(1)
  return self.head(torch.cat([pooled,mask],1))



def drop_one(x,slices,probability=.5,generator=None):
 out=x.clone();mask=out[:,-4:];eligible=mask.sum(1)>1;chosen=(torch.rand(len(x),device=x.device,generator=generator)<probability)&eligible
 if chosen.any():
  scores=torch.rand(mask.shape,device=x.device).masked_fill(mask==0,-1);index=scores.argmax(1)
  for k,(a,b) in enumerate(slices.values()):
   use=chosen&(index==k);out[use,a:b]=0;out[use,out.shape[1]-4+k]=0
 assert bool((out[:,-4:].sum(1)>0).all());return out



def make_weights(y,patterns):
 count=np.zeros((16,3));np.add.at(count,(patterns,y),1);effective=np.zeros_like(count)
 for pattern in range(1,16):
  observed=[b for b in BITS if pattern&b]
  if len(observed)==1:effective[pattern]+=count[pattern]
  else:
   effective[pattern]+=.5*count[pattern]
   for b in observed:effective[pattern-int(b)]+=.5/len(observed)*count[pattern]
 cc=np.bincount(y,minlength=3);cw=np.clip(np.sqrt(len(y)/3/(cc+5)),.25,4);cw/=np.sum(cc*cw)/len(y)
 jw=np.clip(np.sqrt(len(y)/45/(effective+5)),.25,4);jw/=np.sum(effective*jw)/len(y)
 jw_nat=np.clip(np.sqrt(len(y)/45/(count+5)),.25,4);jw_nat/=np.sum(count*jw_nat)/len(y)
 pi=(effective+5)/(effective.sum(1,keepdims=True)+15);pi=np.clip(pi,1e-3,1.)
 assert np.isclose(effective.sum(),len(y))
 report={'class_counts':cc.tolist(),'natural_cell_counts':count.tolist(),'expected_dropout_cell_counts':effective.tolist(),'class_weights':cw.tolist(),'joint_weights_dropout_basis':jw.tolist(),'joint_weights_natural_basis':jw_nat.tolist(),'logit_adjustment_priors':pi.tolist(),'rule':'inverse sqrt counts+5, clip .25-4, mean1; joint_drop uses expected-dropout counts, joint_no_drop uses natural counts; logitadj uses smoothed pattern-class priors (E+5)/(row+15)'}
 return cw.astype(np.float32),jw.astype(np.float32),jw_nat.astype(np.float32),pi.astype(np.float32),report



def preprocess(ep,blocks,train_ids,cfg):
    arrays=[];state={};slices={};offset=0
    for m in MODS:
        tr=blocks[m].loc[train_ids];tr=tr.loc[ep.loc[train_ids,f'm_{m}']==1]
        keep=tr.isna().mean()<=cfg['max_missing_feature_fraction'];tr=tr.loc[:,keep]
        mean=tr.mean().fillna(0);sd=tr.std(ddof=0).fillna(1).clip(lower=1e-6);raw=blocks[m].loc[:,keep]
        a=((raw-mean)/sd).clip(-cfg['z_clip'],cfg['z_clip']).fillna(0).to_numpy(np.float32)
        missing=raw.isna().to_numpy(np.float32);a=np.concatenate([a,missing],1);a[ep[f'm_{m}'].to_numpy()==0]=0
        slices[m]=(offset,offset+a.shape[1]);offset+=a.shape[1];arrays.append(a)
        state[m]={'columns':raw.columns.tolist(),'mean':mean.to_list(),'sd':sd.to_list()}
    mask=ep[[f'm_{m}' for m in MODS]].to_numpy(np.float32);arrays.append(mask)
    return np.concatenate(arrays,axis=1),state,slices



def softmax(z):
    z=z-z.max(1,keepdims=True);e=np.exp(z);return e/e.sum(1,keepdims=True)



def pbar_mat(cell_counts,class_counts):
    n=np.asarray(cell_counts,float);P=np.asarray(class_counts,float);P=P/P.sum()
    num=n+KAPPA*P[None,:];return num/num.sum(1,keepdims=True)



def adjusted(p,pat,pb,tau):
    if tau==0:return p
    lg=np.log(np.clip(p,1e-12,1.));off=np.log(1/3.)-np.log(np.clip(pb[pat],1e-6,1.))
    return softmax(lg+tau*off)

