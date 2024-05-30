import enum

from rotr.action import ActionCode, Priority

class Modifier(enum.Enum):
    MUST = "must"
    SHOULD = "should"

class Action:
    def __init__(self, modifier: str, code: str):
        self._modifier = modifier
        self._code = code
    
    @property 
    def priority(self) -> Priority:
        if self._modifier == Modifier.MUST:
            return Priority.HIGH
        else:
            return Priority.LOW
        
    @property
    def code(self) -> ActionCode:
        try:
            return ActionCode(self._code)
        except:
            return None
    