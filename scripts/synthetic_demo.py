"""CPU-only functional smoke demonstration. NOT an ADNI result or benchmark."""
import json
import numpy as np
import pandas as pd
import torch
from avf import MaskNet, drop_one, make_weights, preprocess, fuse, evaluate, supported_cells

torch.set_num_threads(2)
torch.manual_seed(42)
rng = np.random.default_rng(42)
n = 120
mask = rng.integers(0,2,(n,4))
mask[:,0] = 1
y = np.tile(np.arange(3), n//3)
episodes = pd.DataFrame(mask,columns=["m_mri","m_fdg","m_amy","m_tau"])
blocks = {m:pd.DataFrame(rng.normal(size=(n,6))) for m in ["mri","fdg","amy","tau"]}
x,state,slices = preprocess(episodes,blocks,np.arange(90),{"max_missing_feature_fraction":.2,"z_clip":5})
patterns = mask @ np.array([8,4,2,1])
cw,jw,jw_nat,priors,report = make_weights(y[:90],patterns[:90])
model = MaskNet(slices)
opt = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
for _ in range(2):
    model.train()
    batch = drop_one(torch.tensor(x[:90]),slices)
    effective = (batch[:,-4:].long()*torch.tensor([8,4,2,1])).sum(1)
    weights = torch.tensor(jw)[effective,torch.tensor(y[:90])]
    loss = (torch.nn.functional.cross_entropy(model(batch),torch.tensor(y[:90]),reduction="none")*weights).mean()
    opt.zero_grad(); loss.backward(); opt.step()
model.eval()
with torch.no_grad():
    neural = model(torch.tensor(x[90:])).softmax(1).numpy()
tree_placeholder = rng.dirichlet([2,2,2],len(neural))
mixture = fuse(tree_placeholder,neural,.55)
cells = supported_cells(y[90:],patterns[90:],np.arange(30),min_subjects=1)
print(json.dumps({"demonstration_only":True,"data":"synthetic; no ADNI records", "device":"cpu",
                  "last_training_loss":float(loss.detach()),"probability_rows_valid":bool(np.allclose(mixture.sum(1),1)),
                  "tree_branch":"random probability placeholder, NOT fitted LightGBM",
                  "synthetic_metrics":evaluate(y[90:],mixture,patterns[90:],cells)},indent=2))
