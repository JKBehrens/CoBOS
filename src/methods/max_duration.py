import logging

from control.agent_and_task_states import AgentState, TaskState
from control.jobs import Job
from methods.solver import Solver
import numpy as np

START_AVAILABLE_TASKS = True


class MaxDuration(Solver):

    def __init__(self, job: Job, seed: int):
        super().__init__(job, seed=seed)
        self.rand = np.random.default_rng(seed=self.seed)
        self.task_duration = self.job.task_duration(rand_gen=self.rand)

    @classmethod
    def name(cls) -> str:
        return str(cls).capitalize().split(".")[-1].split("'>")[0]

    def prepare(self):
        return True

    def decide(self, observation_data, current_time):
        decision = {}
        self.update_tasks_status()
        for index, [agent_name, agent_state, agent_current_task, agent_rejection_tasks] in enumerate(observation_data):
            try:
                logging.debug(f'agent {agent_name}, state {agent_state}, task {agent_current_task.id}')
            except AttributeError:
                logging.debug(f'agent {agent_name}, state {agent_state}, task None')

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
                available_tasks.append([task, self.task_duration[agent_name][task.id]])

        if available_tasks:
            available_tasks.sort(key=lambda x: x[1])
            task = available_tasks[0]
            if task[0].universal:
                task[0].agent = [agent_name]
            return task[0]
        return None
