from __future__ import annotations

from minigrid.envs.blockedunlockpickup import BlockedUnlockPickupEnv
from minigrid.envs.crossing import CrossingEnv
from minigrid.envs.distshift import DistShiftEnv
from minigrid.envs.doorkey import DoorKeyEnv
from minigrid.envs.dynamicobstacles import DynamicObstaclesEnv
from minigrid.envs.empty import EmptyEnv
from minigrid.envs.fetch import FetchEnv
from minigrid.envs.fourrooms import FourRoomsEnv
from minigrid.envs.gotodoor import GoToDoorEnv
from minigrid.envs.gotoobject import GoToObjectEnv
from minigrid.envs.keycorridor import KeyCorridorEnv
from minigrid.envs.lavagap import LavaGapEnv
from minigrid.envs.lavafaulty import LavaFaultyEnv
from minigrid.envs.windycity import WindyCityEnv, WindyCityAdvEnv, WindyCity2Env, WindyCitySmallAdv
from minigrid.envs.lavaslippery import LavaSlipperyEnv1, LavaSlipperyCliff, LavaSlipperyHill, LavaSlipperyMaze
from minigrid.envs.GSW_Playground import Playground
from minigrid.envs.adversary_debug import AdversaryDebug
from minigrid.envs.adversary_simple import AdversarySimple
from minigrid.envs.adversarydoorpickup import AdversaryDoorPickup
from minigrid.envs.oscillating_adversaries import OscillatingAdversaries
from minigrid.envs.doubledoor import DoubleDoorEnv
from minigrid.envs.singledoor import SingleDoorEnv
from minigrid.envs.lockedroom import LockedRoom, LockedRoomEnv
from minigrid.envs.memory import MemoryEnv
from minigrid.envs.multiroom import MultiRoom, MultiRoomEnv
from minigrid.envs.obstructedmaze import (
    ObstructedMaze_1Dlhb,
    ObstructedMaze_Full,
    ObstructedMazeEnv,
)
from minigrid.envs.playground import PlaygroundEnv
from minigrid.envs.putnear import PutNearEnv
from minigrid.envs.redbluedoors import RedBlueDoorEnv
from minigrid.envs.unlock import UnlockEnv
from minigrid.envs.unlockpickup import UnlockPickupEnv