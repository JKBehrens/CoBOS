from abc import ABC, abstractmethod
from control.agent_and_task_states import TaskState
from control.jobs import Job


class Solver(ABC):

    @abstractmethod
    def __init__(self, job: Job, seed: int):
        self.job = job
        self.seed = seed
        self.horizon = None

    @classmethod
    def name(cls) -> str:
        return str(cls.__class__)

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
            if set(task.conditions).issubset(tasks_list):
                if not task.state in [TaskState.COMPLETED, TaskState.InProgress]:
                    task.state = TaskState.AVAILABLE
            else:
                task.state = TaskState.UNAVAILABLE
