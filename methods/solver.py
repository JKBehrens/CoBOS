from abc import ABC, abstractmethod
from control.agent_and_task_states import AgentState, TaskState


class Solver(ABC):

    @abstractmethod
    def __init__(self, job):
        self.job = job
        self.horizon = None

    @abstractmethod
    def prepare(self):
        pass

    @abstractmethod
    def decide(self, observation_data, current_time):
        pass

    @abstractmethod
    def get_statistics(self):
        pass

    def update_tasks_status(self):
        """
        Updates the status of tasks based on their dependencies.
        """
        tasks_list = self.job.get_completed_and_in_progress_task_list()
        for task in self.job.task_sequence:
            if len(task.conditions) != 0 and task.state == TaskState.UNAVAILABLE:
                if set(task.conditions).issubset(tasks_list):
                    task.state = TaskState.AVAILABLE