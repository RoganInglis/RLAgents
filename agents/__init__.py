from agents.base_agent import BaseAgent

__all__ = [
    "BaseAgent",
    "DQNAgent"
]


def make_agent(config):
    if config['agent_name'] in __all__:
        return globals()[config['agent_name']](config)
    else:
        raise Exception('The agent name %s does not exist' % config['agent_name'])


def get_agent_class(config):
    if config['model_name'] in __all__:
        return globals()[config['model_name']]
    else:
        raise Exception('The agent name %s does not exist' % config['agent_name'])
