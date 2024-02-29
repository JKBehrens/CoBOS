from control.agent_and_task_states import AgentState, TaskState
from methods.solver import Solver
import numpy as np

START_AVAILABLE_TASKS = True


class MaxDuration(Solver):

    def __init__(self, job, seed):
        super().__init__(job)
        self.horizon = None
        self.rand = np.random.default_rng(seed=seed)
        self.task_duration = self.job.task_duration(rand_gen=self.rand)


    def prepare(self):
        return True

    def decide(self, observation_data, current_time):
        decision = {}
        self.update_tasks_status()
        for index, [agent_name, agent_state, agent_current_task, agent_rejection_tasks] in enumerate(observation_data):
            if agent_state == AgentState.IDLE or agent_state == AgentState.REJECTION or agent_state == AgentState.DONE:
                decision[agent_name] = self.find_task(agent_name, agent_rejection_tasks)
            else:
                decision[agent_name] = None
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
                     (task.agent == agent_name and task.id not in rejection_tasks)):
                available_tasks.append([task, self.task_duration[agent_name][task.id]])

        if available_tasks:
            available_tasks.sort(key=lambda x: x[1])
            task = available_tasks[0]
            if task[0].universal:
                task[0].agent = agent_name
            return task[0]
        return None
