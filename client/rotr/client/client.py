import requests
import enum

from typing import List
from rotr.belief import BeliefCode
from rotr.intention import IntentionCode
from rotr.client.action import Action

class Client:
    def __init__(self, server, port):
        self._server = server
        self._port = port
        self._base_url = f"http://{self._server}:{self._port}"
        print(f"Connecting to {self._base_url}")

    def get_action(self, belief: List[BeliefCode], intention: List[IntentionCode], context="standard") -> str:
        intention = [i.value for i in intention]
        belief = [b.value for b in belief]
        
        data = {
            "context": context,
            "belief": belief,
            "intention": intention
        }
        response = requests.patch(self._base_url, json=data)
        actions = [Action(a) for a in response.json()]
        return actions
    
    def get_all_actions(self, context="standard"):
        params = {
            "context": context
        }
        url = f"{self._base_url}/actions"

        response = requests.get(url, params=params)
        return response.json()
    
    def get_all_beliefs(self, context="standard"):
        params = {
            "context": context
        }
        url = f"{self._base_url}/beliefs"
        
        response = requests.get(url, params=params)
        return response.json()
    
    def get_all_intentions(self, context="standard"):
        params = {
            "context": context
        }
        url = f"{self._base_url}/intentions"
        
        response = requests.get(url, params=params)
        return response.json()