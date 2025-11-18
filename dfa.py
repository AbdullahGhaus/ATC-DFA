class DFA:
    def __init__(self, states, alphabet, transitions, start_state, accept_states):
        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions
        self.start_state = start_state
        self.accept_states = accept_states

    def accepts(self, text):
        current = self.start_state
        for char in text.lower():
            if char in self.alphabet:
                current = self.transitions.get((current, char), None)
                if current is None:
                    return False
                # Return True as soon as we reach an accept state (substring match)
                if current in self.accept_states:
                    return True
        return current in self.accept_states
