import gymnasium as gym
import numpy as np
import random
from functools import reduce
from operator import mul
from moviepy.editor import ImageSequenceClip
from minigrid.core.state import State
from minigrid.core.state import AdversaryState

import itertools
from utils import MiniGridShieldHandler, common_parser
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import Image


#adversaries=(AdversaryState(color='Blue', col=np.int64(8), row=np.int64(2), view=3, carrying='')

def get_all_states( adstate):    
    ans = []

    ans.append(AdversaryState(color=adstate.color, col = adstate.col, row=adstate.row, view = (adstate.view +3) % 4,carrying = adstate.carrying))
        
    ans.append(AdversaryState(color=adstate.color, col = adstate.col, row=adstate.row, view = (adstate.view +1) % 4,carrying = adstate.carrying))
        
    if adstate.view == 0:
        ans.append(AdversaryState(color=adstate.color, col = adstate.col+1, row=adstate.row, view = adstate.view ,carrying = adstate.carrying))
    elif adstate.view == 1:
        ans.append(AdversaryState(color=adstate.color, col = adstate.col, row=adstate.row+1, view = adstate.view,carrying = adstate.carrying))
    elif adstate.view == 2 :
        ans.append(AdversaryState(color=adstate.color, col = adstate.col-1, row=adstate.row, view = adstate.view ,carrying = adstate.carrying))
    elif adstate.view == 3 :
        ans.append(AdversaryState(color=adstate.color, col = adstate.col, row=adstate.row-1, view =adstate.view,carrying = adstate.carrying))
        
    return ans
    
class MiniGridSbShieldingWrapper(gym.core.Wrapper):
    def __init__(self,
                 env,
                 shield_handler : MiniGridShieldHandler,
                 create_shield_at_reset = False,
                 ):
        super().__init__(env)
        self.shield_handler = shield_handler
        self.create_shield_at_reset = create_shield_at_reset

        shield = self.shield_handler.create_shield(env=self.env)
        self.shield = shield
        self.step_count = 0
        self.num_blackouts = 0
        self.n = 10
        self.k = 1
        self.blackout = False
        self.method = 0
        self.use_prob = False
        # 1 - no shield, 2 - random shield, 3 - prev shield, 4 - out shield
   #State(colAgent=np.int64(1), rowAgent=np.int64(5), viewAgent=1, carrying='', adversaries=(AdversaryState(color='Blue', col=np.int64(8), row=np.int64(2), view=3, carrying=''),), balls=(), boxes=(), keys=(), doors=(), lockeddoors=()) 
    
    def create_action_mask(self):
        if not self.blackout:
            try:
                self.pos_ad_loc = [[x] for x in self.env.get_symbolic_state().adversaries]
                #print(self.pos_ad_loc)
                self.prev_sym_state = self.env.get_symbolic_state()
                return self.shield[self.env.get_symbolic_state()]
            except:
                return [0.0] * 3 + [0.0] * 4
        else:
            if self.method == 1:
                return [1.0] * 3 + [1.0] * 4
            elif self.method == 2:
                return self.prev_shield[self.prev_sym_state]
            elif self.method == 3:
                return [random.choice([0.0, 1.0]) for _ in range(7)]
            else:
                pos_ad_locs = []
                for u in self.pos_ad_loc:
                    r = []
                    for v in u:
                        r.extend(get_all_states(v))
                    pos_ad_locs.append(r)
                cross_product = list(itertools.product(*pos_ad_locs))
                self.pos_ad_loc = pos_ad_locs
                curr = self.env.get_symbolic_state()

                poss_new_states = [State(colAgent = curr.colAgent, rowAgent=curr.rowAgent,viewAgent=curr.viewAgent,carrying=curr.carrying, adversaries=tuple(adst)) for adst in cross_product]
                poss_new_states = [x for x in poss_new_states if x in self.shield.keys()]
                poss_shields = [self.shield[state] for state in poss_new_states]
                intersection_shield = [reduce(mul, items) for items in zip(*poss_shields)]
                return intersection_shield
            

    
            
    def reset(self, *, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)

        if self.create_shield_at_reset:
            shield = self.shield_handler.create_shield(env=self.env)
            self.shield = shield
        return obs, infos

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        info["no_shield_action"] = not self.shield.__contains__(self.env.get_symbolic_state())
        self.step_count += 1
        if(self.step_count % self.n == 0):
            self.num_blackouts = 0
        
        self.blackout = False

        if(random.random() < self.k/self.n):
            if(self.use_prob):
                self.blackout =  True
                self.num_blackouts += 1
            else:
                if(self.num_blackouts<self.k and self.step_count>1):
                    self.blackout = True
                    self.num_blackouts+=1

        return obs, rew, done, truncated, info

def parse_sb3_arguments():
    parser = common_parser()
    args = parser.parse_args()

    return args

class ImageRecorderCallback(BaseCallback):
    def __init__(self, eval_env, render_freq, n_eval_episodes, evaluation_method, log_dir, deterministic=True, verbose=0):
        super().__init__(verbose)

        self._eval_env = eval_env
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic
        self._evaluation_method = evaluation_method
        self._log_dir = log_dir

    def _on_training_start(self):
        image = self.training_env.render(mode="rgb_array")
        self.logger.record("trajectory/image", Image(image, "HWC"), exclude=("stdout", "log", "json", "csv"))

    def _on_step(self) -> bool:
        #if self.n_calls % self._render_freq == 0:
        #    self.record_video()
        return True

    def _on_training_end(self) -> None:
        self.record_video()

    def record_video(self) -> bool:
        screens = []
        def grab_screens(_locals, _globals) -> None:
            """
            Renders the environment in its current state, recording the screen in the captured `screens` list

            :param _locals: A dictionary containing all local variables of the callback's scope
            :param _globals: A dictionary containing all global variables of the callback's scope
            """
            screen = self._eval_env.render()
            screens.append(screen)
        self._evaluation_method(
            self.model,
            self._eval_env,
            callback=grab_screens,
            n_eval_episodes=self._n_eval_episodes,
            deterministic=self._deterministic,
        )

        clip = ImageSequenceClip(list(screens), fps=3)
        clip.write_gif(f"{self._log_dir}/{self.n_calls}.gif", fps=3)
        return True


class InfoCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.sum_goal = 0
        self.sum_lava = 0
        self.sum_collisions = 0
        self.sum_opened_door = 0
        self.sum_picked_up = 0
        self.no_shield_action = 0

    def _on_step(self) -> bool:
        infos = self.locals["infos"][0]
        if infos["reached_goal"]:
            self.sum_goal += 1
        if infos["ran_into_lava"]:
            self.sum_lava += 1
        self.logger.record("info/sum_reached_goal", self.sum_goal)
        self.logger.record("info/sum_ran_into_lava", self.sum_lava)
        if "collision" in infos:
            if infos["collision"]:
                self.sum_collisions += 1
            self.logger.record("info/sum_collision", self.sum_collisions)
        if "opened_door" in infos:
            if infos["opened_door"]:
                self.sum_opened_door += 1
            self.logger.record("info/sum_opened_door", self.sum_opened_door)
        if "picked_up" in infos:
            if infos["picked_up"]:
                self.sum_picked_up += 1
            self.logger.record("info/sum_picked_up", self.sum_picked_up)
        if "no_shield_action" in infos:
            if infos["no_shield_action"]:
                self.no_shield_action += 1
            self.logger.record("info/no_shield_action", self.no_shield_action)
        return True
