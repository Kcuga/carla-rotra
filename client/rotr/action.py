import carla
import enum

from abc import ABC, abstractmethod
from typing import List

from rotr.client.action import BeliefCode as Belief

class ActionCode(enum.Enum):
    SLOW_DOWN = "slow_down"
    FOLLOW_SIGN = "follow_sign"
    STOP = "stop"

class Priority(enum.Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2

class Action(ABC):
    def __init__(self, priority: Priority=Priority.LOW):
        self._priority = priority

    def get_priority(self):
        return self._priority
    
    def set_priority(self, priority: Priority):
        self._priority = priority

    @abstractmethod
    def get_control(self):
        pass

    @abstractmethod
    def apply(self, vehicle: carla.Vehicle):
        pass


class ControlAction(Action):
    def __init__(self, priority: Priority=Priority.LOW, throttle: float=0.0, steer: float=0.0, brake: float=0.0, hand_brake: bool=False):
        super().__init__(priority)
        self._control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake, hand_brake=hand_brake)

    def get_control(self):
        return self._control

    def apply(self, vehicle: carla.Vehicle):
        vehicle.apply_control(self._control)


# class PlannerAction(Action):
#     def __init__(self, planner: carla.LocalPlanner):
#         super().__init__()
#         self._local_planner = planner

#     def get_control(self):
#         pass

#     def apply(self, vehicle: carla.Vehicle):
#         pass

class ActionMapper:
    def __init__(self):
        self._mapping = {
            ActionCode.STOP: ControlAction(throttle=0.0, brake=1.0, priority=Priority.HIGH),
            ActionCode.SLOW_DOWN: ControlAction(throttle=0.0, brake=.2),
        }

    def get(self, code: ActionCode) -> Action:
        return self._mapping.get(code, None)
    
    def map(self, actions_views: List[Action]) -> List[Action]:
        actions = []
        for av in actions_views:
            if av.code is None:
                continue
            
            action = self.get(av.code)
            if action:
                action.set_priority(av.priority)
                actions.append(action)
        
        actions.sort(key=lambda action: action._priority.value)
        return actions

    def __contains__(self, code: ActionCode):
        return code in self._mapping
    
    def __getitem__(self, code: ActionCode):
        return self._mapping.get(code, None)


