from base import top_state, sub_state

from .initialization import initialization
from .execution import execution
from .analysis import analysis

class FSM(top_state):
    def __init__(self):
        super(FSM, self).__init__()

        # Substates
        self.SetSubstate(initialization = initialization(), 
                         execution = execution(),
                         analysis = analysis())

    def BuildControlFlow(self):
        self.initialization >= self.execution >= self.analysis
        return self.initialization