import carla

from abc import ABC, abstractmethod
from typing import List, Tuple

from rotr.belief import BeliefCode as Belief
from rotr.intention import IntentionCode as Intention
from rotr.utils import get_traffic_light_trigger_location, is_within_distance

class Observer(ABC):
    def __init__(self, vehicle):
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
    
    @abstractmethod
    def observe(self) -> List[str]:
        pass


class TrafficLightObserver(Observer):
    def __init__(self, vehicle, map = None, max_distance = 5.0, context = "standard"):
        super().__init__(vehicle)
        self._context = context
        self._max_distance = max_distance
        self._world = self._vehicle.get_world()

        if map:
            if isinstance(map, carla.Map):
                self._map = map
            else:
                self._map = self._world.get_map()
        else:
            self._map = self._world.get_map()

        self._lights_list = self._world.get_actors().filter("*traffic_light*")
        self._lights_map = {}
        self._last_traffic_light = None

    def observe(self) -> Tuple[str, List[str], List[str]]:
        context = self._context
        belief = []
        intention = []

        # If at traffic light
        if self._last_traffic_light:
            belief.append(Belief.AT_TRAFFIC_LIGHT)
            intention.append(Intention.SET_OFF)

            if self._last_traffic_light.state != carla.TrafficLightState.Red:
                self._last_traffic_light = None
            else:
                belief.append(Belief.LIGHT_NOT_GREEN)
        # Check if approaching traffic light
        else:
            # Get vehicle location and next waypoint
            vehicle_location = self._vehicle.get_location()
            vehicle_waypoint = self._map.get_waypoint(vehicle_location)

            for traffic_light in self._lights_list:
                if traffic_light.id in self._lights_map:
                    trigger_wp = self._lights_map[traffic_light.id]
                else:
                    trigger_location = get_traffic_light_trigger_location(traffic_light)
                    trigger_wp = self._map.get_waypoint(trigger_location)
                    self._lights_map[traffic_light.id] = trigger_wp

                # Ignore traffic lights that are too far away
                if trigger_wp.transform.location.distance(vehicle_location) > self._max_distance:
                    continue

                # Ignore traffic lights that are not on the same road
                if trigger_wp.road_id != vehicle_waypoint.road_id:
                    continue

                ve_dir = vehicle_waypoint.transform.get_forward_vector()
                wp_dir = trigger_wp.transform.get_forward_vector()
                dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

                # Ignore traffic lights that are not in front of the vehicle
                if dot_ve_wp < 0:
                    continue

                # Check if the traffic light is within the distance and angle
                if is_within_distance(trigger_wp.transform, self._vehicle.get_transform(), self._max_distance, [0, 90]):
                    self._last_traffic_light = traffic_light
                    
                    # Vehicle is approaching a traffic light
                    intention.append(Intention.APPROACHING_TRAFFIC_LIGHT)

                    # Check traffic light state
                    if traffic_light.state == carla.TrafficLightState.Red:
                        belief.append(Belief.LIGHT_RED)
                    elif traffic_light.state == carla.TrafficLightState.Yellow:
                        belief.append(Belief.LIGHT_AMBER)

                        #TODO: Check if can safely stop at white line
                    else:
                        belief.append(Belief.LIGHT_GREEN)
                    
                    break

        return context, belief, intention

class WorldObserver(Observer):

    def __init__(self, vehicle, map=None, context="standard"):
        super().__init__(vehicle)
        self._context = context
        self._world = self._vehicle.get_world()

        if map:
            if isinstance(map, carla.Map):
                self._map = map
            else:
                self._map = self._world.get_map()
        else:
            self._map = self._world.get_map()

        self._observers = [
            TrafficLightObserver(self._vehicle, self._map, context=self._context)
        ]

    def observe(self) -> Tuple[str, List[str], List[str]]:
        context = self._context
        belief = []
        intention = []

        for observer in self._observers:
            _, b, i = observer.observe()
            belief.extend(b)
            intention.extend(i)

        return context, belief, intention