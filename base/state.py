from .dispatcher import dispatcher
import inspect

# Base class
class state:
    def __init__(self):
        super().__setattr__('dataflow', {})
        super().__setattr__('nextState', [None])

    # Push multiple variables in the dataflow 
    def AddToDataflow(self, **vars):
        self.dataflow.update(vars)
    
    # Work is done here
    def Action(self):          
        return None      # return is the next state

    def GetStateByClass(self, cls):
        for state in self.nextState:
            if isinstance(state, cls):
                return state

    # Update or add a varible to the dataflow
    def __setattr__(self, name, value):
        self.dataflow[name] = value
    
    # Varible in the data flow can be used like a member variable
    def __getattr__(self, name):
        if name in self.dataflow:
            return self.dataflow[name]

        raise AttributeError('Variable %s not found in the current scope'%name)

    # Push dataflow in the state and execute
    def __call__(self, dataflow):
        super().__setattr__('dataflow', dataflow)
        nextState = self.Action()          # Dataflow is supposed be modified in Action
        return nextState, self.dataflow

    # Overload operator >= to create a state chain
    # Example (a loop of states):
    # self.substate1 >= self.substate2 >= self.substate3 >= self.substate1
    # self.substate3 >= self.endstate
    def __ge__(self, other):
        if self.nextState[0] == None:
            super().__setattr__('nextState', [other])
        else:
            if not other in self.nextState:
                self.nextState.append(other)
        return other
    
    # This method is not recommended for its low effiency
    # Example:
    # x = 1
    # z = 3
    # self << x << ('y', 2) << z
    # The above snippet will create three new items in the dataflow
    # The last line equal to: self.AddToDataflow(x=x, y=2, z=z)
    def __lshift__(self, other):
        if not 'varNames' in self.__dict__:
            # cache all variable names in a line which will be added to the dataflow
            source = inspect.stack()[1].code_context[0]
            varNamesRaw = source.split('<<')[1:]
            self.__dict__['varNames'] =  [name.strip() for name in varNamesRaw]

        if type(other) == tuple and len(other) == 2 and type(other[0]) == str:
            self.dataflow[other[0]] = other[1]
        else:
            self.dataflow[self.varNames[0]] = other

        self.varNames.pop(0)
        if len(self.varNames) == 0:
            del self.__dict__['varNames']

        return self

# A wrapper containing multiple substates
class top_state(state):
    def __init__(self):
        super(top_state, self).__init__()

    def Action(self):
        entryState = self.BuildControlFlow()

        localDispatcher = dispatcher(entryState, **self.dataflow)
        subDataflow = localDispatcher.run()

        nextState, finalDataflow = self.GetNextState(subDataflow)
        self.AddToDataflow(**finalDataflow)
        return nextState

    def BuildControlFlow(self) -> state: # return is the entry state
        pass                            
    
    def GetNextState(self, subDataflow) -> state:
        return self.nextState[0], subDataflow
    
    def SetSubstate(self, **substateDict):
        for name, value in substateDict.items():
            super().__dict__[name] = value

class sub_state(state):
    pass
