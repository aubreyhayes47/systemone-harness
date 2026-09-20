from systemone_harness import Action, SystemOneHarness


class TinyEnvironment:
    goal = "Get safely across the room."
    done = False

    def __init__(self):
        self.position = 0

    def observe(self):
        return {"position": self.position, "target": 3}

    def legal_actions(self):
        return [
            Action("forward", "Move one step toward the target."),
            Action("wait", "Stay where you are."),
        ]

    def execute(self, action: Action):
        if action.id == "forward":
            self.position += 1
        self.done = self.position >= 3


env = TinyEnvironment()

with SystemOneHarness.from_env() as harness:
    while not env.done:
        decision = harness.decide(
            goal=env.goal,
            state=env.observe(),
            actions=env.legal_actions(),
        )
        print(decision.action.id, dict(decision.probabilities))
        env.execute(decision.action)
