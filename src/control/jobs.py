"""
    Job class contains the variables to describe the job and the functions needed to define it.
    Task class contains the variables needed to describe the task.

    @author: Marina Ionova, student of Cybernetics and Robotics at the CTU in Prague
    @contact: marina.ionova@cvut.cz
"""
from control.representation import JobDescription, Task, Action 
from inputs import RandomCase
from control.agent_and_task_states import TaskState
import numpy as np
import logging


class Job:
    """
    A class representing a job consisting of multiple tasks.

    :param case: Input case for generating job description.
    :type case: str
    """

    def __init__(self, case: int, seed: int, random_case_param: RandomCase = RandomCase()):
        self.case = case
        self.seed = seed
        from inputs import case_generator
        self.job_description: JobDescription = JobDescription(tasks=case_generator.set_input(self.case, self.seed,random_case_param))
        self.task_sequence: list[Task] = self.job_description.tasks
        self.in_progress_tasks:list[int] = []
        self.completed_tasks: list[int] = []
        self.agents: list[str] = ["Human", "Robot"]
        self.task_number = len(self.task_sequence)
        self.predicted_makespan = None

    def __str__(self):
        """
        Returns a string representation of the job's task list.
        """
        logging.info("Task list")
        logging.info("____________________________")
        s = ""
        for agent in self.agents:
            for task in self.task_sequence:
                if agent in task.agent:
                    s += task.__str__() + "\n"
        logging.info("____________________________")

        return s

    def __repr__(self) -> str:
        return self.__str__()

    def progress(self):
        """
        Returns the percentage of completed tasks in the job.
        """
        completed_tasks = sum( 1 for task in self.task_sequence if task.state == TaskState.COMPLETED)
        return round((completed_tasks / self.task_number) * 100, 2)

    def get_current_makespan(self) -> int:
        """
        Returns the current makespan of the job.
        """
        return max(task.finish[0] for task in self.task_sequence)

    def get_completed_and_in_progress_task_list(self) -> list[int]:
        output: list[int] = []
        for task in self.task_sequence:
            if task.state == TaskState.COMPLETED or task.state == TaskState.InProgress:
                output.append(task.id)
        return output

    def task_duration(self,  rand_gen: np.random.Generator | None = None,  seed: int | None =None):
        if rand_gen is None:
            rand_gen = np.random.default_rng(seed)
        assert isinstance(rand_gen, np.random.Generator)

        task_duration = {0: {}, 1: {}, 'Robot': {}, 'Human': {}}
        for task in self.task_sequence:
            robot = task.get_duration(rand_gen=rand_gen)
            task_duration['Robot'][task.id] = robot
            task_duration[0][task.id] = robot
            human = task.get_duration(rand_gen=rand_gen)
            task_duration[1][task.id] = human
            task_duration['Human'][task.id] = human

        return task_duration

    def get_task_idx(self, task: Task):
        """
        Returns the index of a specified task in the job's task sequence.

        :param task: Task to find index of.
        :type task: Task
        :return: Index of specified task.
        :rtype: int
        """
        assert task.id == self.task_sequence.index(task)
        return task.id

    def refresh_completed_task_list(self, task_id:int):
        """
        Adds a completed task to the job's completed task list and removes it from the in-progress task list.

        :param task_id: ID of completed task.
        :type task_id: int
        """
        self.completed_tasks.append(task_id)
        self.in_progress_tasks.remove(task_id)

    def get_universal_task_number(self):
        return sum(1 for task in self.task_sequence if task.universal)

    def change_agent(self, task_id: int, new_agent_name: str):
        # Find the task with the given task_id
        matching_task = next((task for task in self.task_sequence if task.id == task_id), None)

        if matching_task:
            matching_task.agent = [new_agent_name]

    def validate(self):
        valid = True
        messages: list[str] = []
        for task in self.task_sequence:
            if not isinstance(task.finish, list) or not len(task.finish) == 4:
                messages.append(f"Task {task.id} has no timing")
                valid =False
            else:
                start: int = task.finish[0] - task.finish[2] - task.finish[3]
                if not isinstance(task.start, int) or task.start > task.finish[0]- sum(task.finish[1:]):
                    messages.append(f"Task {task.id} has inconsistent timing. task.tart: {task.start} and task.finish: {task.finish}.")
                    valid = False
                ends: list[int] = []
                for dep in task.conditions:
                    assert self.task_sequence[dep].id == dep
                    assert isinstance(self.task_sequence[dep].finish, list) 
                    assert len(self.task_sequence[dep].finish) == 4
                    ends.append(self.task_sequence[dep].finish[0] - self.task_sequence[dep].finish[3]) # type: ignore

                if len(ends) == 0:
                    continue
                if not start >= max(ends):
                    valid =False
                    messages.append(f"dependency graph violation for task {task.id}.")
        if valid:
            logging.info("The solution is valid. No dependency violation")
            return True
        else:
            for msg in messages:
                logging.error(msg)
        return valid

