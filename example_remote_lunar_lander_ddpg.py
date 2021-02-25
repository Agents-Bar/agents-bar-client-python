from ai_traineree.env_runner import EnvRunner
from ai_traineree.tasks import GymTask
from ai_traineree_client import RemoteAgent


if __name__ == "__main__":

    task = GymTask('LunarLanderContinuous-v2')

    agent = RemoteAgent(
        state_size=task.state_size, action_size=task.action_size,
        url='localhost', agent_model="ddpg", agent_name="ddpg_test_3", description="Nothing special",
        device='cpu')

    env_runner = EnvRunner(task, agent)

    scores = env_runner.run(reward_goal=100, max_episodes=50, force_new=True, eps_decay=0.95)
    env_runner.interact_episode(render=True)
