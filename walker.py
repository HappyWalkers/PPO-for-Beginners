from dm_control import suite,viewer
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

class uth_t(nn.Module):
    def __init__(self,xdim,udim,
                 hdim=32,fixed_var=True):
        super().__init__()
        self.xdim,self.udim = xdim, udim
        self.fixed_var=fixed_var

        self.actor = nn.Sequential(nn.Linear(xdim,hdim),nn.ReLU(),nn.Linear(hdim,hdim),nn.ReLU(),nn.Linear(hdim,udim*2))

    def forward(self,x):
        # assert type(x) is th.Tensor, f"x must be a torch.Tensor, got x={x}"
        if type(x) is not th.Tensor:
            x = x.observation
            x=np.array(x['orientations'].tolist()+[x['height']]+x['velocity'].tolist())
            x = th.tensor(x)
            x = x.float()
        mu,log_std = self.actor(x).chunk(2,dim=-1)
        if self.fixed_var:
            log_std = th.ones_like(log_std)*log_std.mean()
        std = th.exp(log_std)
        return mu.detach().cpu().numpy()

def rollout(e,uth,T=1000):
    """
    e: environment
    uth: controller
    T: time-steps
    """

    traj=[]
    t=e.reset()
    x=t.observation
    x=np.array(x['orientations'].tolist()+[x['height']]+x['velocity'].tolist())
    for _ in range(T):
        with th.no_grad():
            u,_=uth(th.from_numpy(x).float().unsqueeze(0))
        r = e.step(u.numpy())
        x=r.observation
        xp=np.array(x['orientations'].tolist()+[x['height']]+x['velocity'].tolist())

        t=dict(xp=xp,r=r.reward,u=u,d=r.last())
        traj.append(t)
        x=xp
        if r.last():
            break
    return traj

"""
Setup walker environment
"""
r0 = np.random.RandomState(42)
e = suite.load('walker', 'walk',
                 task_kwargs={'random': r0})
U=e.action_spec();udim=U.shape[0];
X=e.observation_spec();xdim=14+1+9;

# #Visualize a random controller
# def u(dt):
#     return np.random.uniform(low=U.minimum,
#                              high=U.maximum,
#                              size=U.shape)
# viewer.launch(e,policy=u)

# Example rollout using a network
uth=uth_t(xdim,udim)
# traj=rollout(e,uth)
viewer.launch(e,policy=uth)