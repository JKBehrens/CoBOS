from control.agent_and_task_states import AgentState, TaskState
from methods.solver import Solver
import numpy as np

START_AVAILABLE_TASKS = True


class RandomAllocation(Solver):

    def __init__(self, job, seed):
        super().__init__(job, seed)
        self.horizon = None
        self.rand = np.random.default_rng(seed=seed)

    @classmethod
    def name(cls) -> str:
        return str(cls).capitalize().split(".")[-1].split("'>")[0]

    def prepare(self):
        return True

    def decide(self, observation_data, current_time):
        decision = {}
        self.update_tasks_status()
        for index, [agent_name, agent_state, agent_current_task, agent_rejection_tasks] in enumerate(observation_data):
            if agent_state == AgentState.REJECTION:
                agent_current_task.agent = [agent_name for agent_name in self.job.agents]

            if agent_state == AgentState.IDLE or agent_state == AgentState.REJECTION or agent_state == AgentState.DONE:
                decision[agent_name] = self.find_task(agent_name, agent_rejection_tasks)
            else:
                decision[agent_name] = None

            if decision[agent_name] is not None:
                decision[agent_name].state = TaskState.ASSIGNED
        return decision

    def get_statistics(self):
        pass

    def find_task(self, agent_name, agent_rejection_tasks):
        available_tasks = []
        if len(agent_rejection_tasks) != 0:
            rejection_tasks = [task_id for task_id in agent_rejection_tasks]
        else:
            rejection_tasks = agent_rejection_tasks
        for task in self.job.task_sequence:
            if task.state == TaskState.AVAILABLE and \
                    ((task.universal and task.id not in rejection_tasks) or
                     (agent_name in task.agent and task.id not in rejection_tasks)):
                available_tasks.append(task)

        if available_tasks:
            task = self.rand.choice(available_tasks, size=1)[0]
            if task.universal:
                task.agent = [agent_name]
            return task
        return None
