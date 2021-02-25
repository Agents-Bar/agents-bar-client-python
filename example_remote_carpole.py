from ai_traineree.env_runner import EnvRunner
# from ai_traineree.loggers import TensorboardLogger
from ai_traineree.tasks import GymTask
from ai_traineree_client import RemoteAgent


if __name__ == "__main__":

    task = GymTask('CartPole-v1')

    agent = RemoteAgent(
        state_size=task.state_size, action_size=task.action_size,
        url='localhost', agent_model="dqn", agent_name="dqn_test_2", description="Nothing special",
        n_steps=3, device='cpu',
    )

    # env_runner = EnvRunner(task, agent, data_logger=data_logger, seed=seed)
    env_runner = EnvRunner(task, agent)

    scores = env_runner.run(reward_goal=100, max_episodes=20, force_new=True, eps_decay=0.98)
    env_runner.interact_episode(render=True)
