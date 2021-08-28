from gym.envs.registration import register

register(
    id='array_escape-v0',
    entry_point='array_escape.envs:ArrayEscapeEnv'
)
register(
    id='array_escape-v1',
    entry_point='array_escape.envs:ArrayEscapeEnvV1'
)

register(
    id='array_escape-v2',
    entry_point='array_escape.envs:ArrayEscapeEnvV2'
)