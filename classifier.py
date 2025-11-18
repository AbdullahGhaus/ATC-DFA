from dfa import DFA

def build_keyword_dfa(keyword):
    keyword = keyword.lower()
    m = len(keyword)
    states = list(range(m + 1))
    alphabet = set("abcdefghijklmnopqrstuvwxyz ")

    transitions = {}

    for s in states:
        for c in alphabet:
            # default transition
            transitions[(s, c)] = 0

    # Build transitions like KMP prefix automaton
    for i in range(m):
        transitions[(i, keyword[i])] = i + 1

    # Add failure transitions
    for s in range(1, m):
        prev = transitions[(s, keyword[s - 1])]
        for c in alphabet:
            if c != keyword[s]:
                transitions[(s, c)] = transitions[(prev, c)]

    return DFA(
        states=states,
        alphabet=alphabet,
        transitions=transitions,
        start_state=0,
        accept_states=[m]  # accept final state
    )


def classify_message(dfalist, text):
    text = text.lower()
    for dfa in dfalist:
        if dfa.accepts(text):
            return "spam"
    return "ham"
