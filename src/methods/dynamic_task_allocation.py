from control.agent_and_task_states import AgentState, TaskState
from control.jobs import Job
from methods.solver import Solver
import numpy as np

START_AVAILABLE_TASKS = True


class DynamicAllocation(Solver):

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
        available_tasks = self.are_there_available_tasks()
        if len(available_tasks) != 0:
            for index, [
                agent_name,
                agent_state,
                agent_current_task,
                agent_rejection_tasks,
            ] in enumerate(observation_data):
                if agent_state == AgentState.REJECTION:
                    agent_current_task.agent = [
                        agent_name for agent_name in self.job.agents
                    ]
                if (
                    agent_state == AgentState.IDLE
                    or agent_state == AgentState.REJECTION
                    or agent_state == AgentState.DONE
                ):
                    decision[agent_name] = self.find_task(
                        agent_name, agent_rejection_tasks
                    )
                else:
                    decision[agent_name] = None
                if decision[agent_name] is not None:
                    decision[agent_name].state = TaskState.ASSIGNED
        else:
            decision = {agent_name: None for agent_name in self.job.agents}
        return decision

    def are_there_available_tasks(
        self,
        rejection_tasks: list[int] = None,
        agent: str = None,
        universal: bool = False,
    ):
        available_tasks = []
        if agent is None:
            for task in self.job.task_sequence:
                if task.state is TaskState.AVAILABLE:
                    available_tasks.append(task)
        else:
            for task in self.job.task_sequence:
                if (
                    task.state is TaskState.AVAILABLE
                    and task.universal == universal
                    and agent in task.agent
                    and task.id not in rejection_tasks
                ):
                    available_tasks.append(task)
        return available_tasks

    def get_statistics(self):
        pass

    def find_task(self, agent_name, agent_rejection_tasks):
        available_tasks_for_agent = self.are_there_available_tasks(
            agent_rejection_tasks, agent_name, universal=False
        )
        if len(available_tasks_for_agent) != 0:
            durations = np.array(
                [
                    self.task_duration[agent_name][task.id][0]
                    for task in available_tasks_for_agent
                ]
            )
            min_index = np.argmin(durations)
            return available_tasks_for_agent[min_index]

        available_universal_tasks = self.are_there_available_tasks(
            agent_rejection_tasks, agent_name, universal=True
        )
        if len(available_universal_tasks) != 0:
            coworker_name = self.job.agents[self.job.agents.index(agent_name) - 1]
            time_advantage = np.array(
                [
                    self.task_duration[coworker_name][task.id][0]
                    - self.task_duration[agent_name][task.id][0]
                    for task in available_universal_tasks
                ]
            )
            max_index = np.argmax(time_advantage)
            available_universal_tasks[max_index].agent = [agent_name]
            return available_universal_tasks[max_index]
        return None
