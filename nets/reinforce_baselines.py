class Baseline(object):

    def eval(self, x, c):
        raise NotImplementedError("Override this method")

    def get_learnable_parameters(self):
        return []

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass



class CriticBaseline(Baseline):

    def __init__(self, critic):
        super(Baseline, self).__init__()
        self.critic = critic

    def eval(self, x, solutions):
        v = self.critic(x, solutions)
        # Detach v since actor should not backprop through baseline, only for loss
        return v.detach().squeeze(), v.squeeze()

    def get_learnable_parameters(self):
        return list(self.critic.parameters())

    def state_dict(self):
        return {
            'critic': self.critic.state_dict()
        }

    def load_state_dict(self, state_dict):
        critic_state_dict = state_dict.get('critic', {})
        if not isinstance(critic_state_dict, dict):  # backwards compatibility
            critic_state_dict = critic_state_dict.state_dict()
        self.critic.load_state_dict({**self.critic.state_dict(), **critic_state_dict})
