# Import necessary libraries
import random
import threading
import carla
import socket
import time

import rotr
from rotr.agent import Agent

PORT = 6060
SERVER = socket.gethostbyname(socket.gethostname())
ADDR = (SERVER, PORT)

HEADER = 8
FORMAT = 'utf-8'


def update_spectator_transform(vehicle, spectator, stop_event):
    while not stop_event.is_set():
        try:
            # Retrieve the vehicle's transform and set the spectator's transform accordingly
            if vehicle.is_alive:
                spectator_transform = vehicle.get_transform()
                spectator_transform.location.z = 3  # Set the z-coordinate to 3
                spectator.set_transform(spectator_transform)
                time.sleep(0.01)  # Adjust the update frequency as needed
            else:
                break
        except RuntimeError:
            # Handle the case where the vehicle is destroyed
            break


# Main function to run CARLA simulation
def run_carla_simulation():

    try:
        # Connect to CARLA server
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)
        print("Connected to CARLA server!")

        # Get the world and blueprint library
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()

        # Spawn a vehicle
        # Choose a random spawn point
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        vehicle_bp = blueprint_library.find('vehicle.ford.mustang')
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        print("Vehicle spawned!")

        time.sleep(1)

        # Spawn a spectator directly above the car
        spectator = world.get_spectator()
        spectator_transform = vehicle.get_transform()
        spectator_transform.location.z = 3  # Set the z-coordinate to 3
        spectator.set_transform(spectator_transform)

        # Create a thread stop event
        stop_event = threading.Event()
        # Start a thread to continuously update the spectator's transform
        spectator_thread = threading.Thread(target=update_spectator_transform, args=(vehicle, spectator, stop_event))
        spectator_thread.start()

        agent = Agent(vehicle, SERVER, PORT, debug=True)
        agent.set_autopilot(True)

        print("Set autopilot to True!")

        # Simulate for a while
        for _ in range(60):
            time.sleep(1)

            agent.run_step()
    finally:
        # Set the stop event to stop the spectator thread
        stop_event.set()

        # Cleanup
        if 'vehicle' in locals():
            vehicle.destroy()
            print("Vehicle destroyed!")


if __name__ == '__main__':
    run_carla_simulation()
