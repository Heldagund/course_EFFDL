class dispatcher:
    def __init__(self, entryState, **globalVars):
        self.currentState = entryState

        self.dataflow = {}
        for name, value in globalVars.items():
            self.dataflow[name] = value

    def run(self):
        while not self.currentState == None:
            print('Enter state:', self.currentState.__class__.__name__)
            self.nextState, self.dataflow = self.currentState(self.dataflow)
            print('Leave state:', self.currentState.__class__.__name__)
            self.currentState = self.nextState
        return self.dataflow