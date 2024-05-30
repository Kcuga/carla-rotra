import carla

from typing import List

from rotr.client.client import Client
from rotr.observer import WorldObserver
from rotr.action import ActionMapper, Action

class Agent:
    def __init__(self, vehicle, server, port, context="standard", map=None, grp=None, debug=False):
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        if map is not None:
            if isinstance(map, carla.Map):
                self._map = map
            else:
                self._map = self._world.get_map()
        else:
            self._map = self._world.get_map()

        # ROTR
        self._context = context
        self._rotr_client = Client(server, port)
        self._observer = WorldObserver(self._vehicle, map=self._map, context=self._context)
        self._action_mapper = ActionMapper()
    
        # Autopilot override information
        self._cooldown = 5
        self._autopilot_override = False
        self._wait_count = 0

        self._debug = debug


    def set_autopilot(self, enable):
        self._vehicle.set_autopilot(enable)

    def _get_actions(self) -> List[Action]:
        c, b, i = self._observer.observe()
        if self._debug:
            print("Context:", c)
            print("Belief:", b)
            print("Intention:", i)
        a = self._rotr_client.get_action(b, i, context=c)
        
        if self._debug:
            print("Actions:", a)

        return self._action_mapper.map(a)

    def run_step(self):
        if self._debug:
            print("Running step...")

        if self._autopilot_override:
            if self._debug:
                print(f"Resuming autopilot in {self._cooldown - self._wait_count} seconds.")
            
            self._wait_count += 1
            if self._wait_count == self._cooldown:
                self._wait_count = 0
                self._autopilot_override = False
                self.set_autopilot(True)
            return

        actions = self._get_actions()

        if len(actions) == 0:
            return
        else:
            if self._debug:
                print("Overriding autopilot...")
            
            self.set_autopilot(False)
            self._autopilot_override = True

            for action in actions:
                action.apply(self._vehicle)