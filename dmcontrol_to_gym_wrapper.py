from dm_control import suite,viewer
import numpy as np

class DmcontrolToGymWrapper:
    def __init__(self, env) -> None:
        self.env = env

    def step(self, action):
        r = self.env.step(action)

        x=r.observation
        xp=np.array(x['orientations'].tolist()+[x['height']]+x['velocity'].tolist())
        obs = xp

        reward = r.reward

        done = r.last()
        return obs, reward, done, r
    
    def reset(self):
        t=self.env.reset()	
        x=t.observation
        x=np.array(x['orientations'].tolist()+[x['height']]+x['velocity'].tolist())
        obs = x
        return obs
    
    def observation_space(self):
        observation_spec = self.env.observation_spec()
        obs_dim = 0
        for key in observation_spec:
            obs_dim += observation_spec[key].shape[0] if len(observation_spec[key].shape) > 0 else 1
        return obs_dim
    
    def action_space(self):
        action_spec = self.env.action_spec()
        return action_spec.shape[0]