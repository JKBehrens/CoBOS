"""
    Job class contains the variables to describe the job and the functions needed to define it.
    Task class contains the variables needed to describe the task.

    @author: Marina Ionova, student of Cybernetics and Robotics at the CTU in Prague
    @contact: marina.ionova@cvut.cz
"""
from pydantic import BaseModel
from inputs import RandomCase
from control.agent_and_task_states import TaskState
from typing import List, Optional
import numpy as np
import logging
import copy


class Job:
    """
    A class representing a job consisting of multiple tasks.

    :param case: Input case for generating job description.
    :type case: str
    """

    def __init__(self, case: int, seed: int, randon_case_param: RandomCase = None):
        self.case = case
        self.seed = seed
        from inputs import case_generator
        self.job_description = case_generator.set_input(self.case, self.seed,randon_case_param)
        self.task_sequence: List[Task] = self.job_description
        self.in_progress_tasks = []
        self.completed_tasks = []
        self.agents = ["Human", "Robot"]
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

    def get_current_makespan(self):
        """
        Returns the current makespan of the job.
        """
        return max(task.finish[0] for task in self.task_sequence)

    def get_completed_and_in_progress_task_list(self):
        output = []
        for task in self.task_sequence:
            if task.state == TaskState.COMPLETED or task.state == TaskState.InProgress:
                output.append(task.id)
        return output

    def task_duration(self,  rand_gen: Optional[np.random.Generator] = None,  seed=None):
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

    def get_task_idx(self, task):
        """
        Returns the index of a specified task in the job's task sequence.

        :param task: Task to find index of.
        :type task: Task
        :return: Index of specified task.
        :rtype: int
        """
        return self.task_sequence.index(task)

    def refresh_completed_task_list(self, task_id):
        """
        Adds a completed task to the job's completed task list and removes it from the in-progress task list.

        :param task_id: ID of completed task.
        :type task_id: int
        """
        self.completed_tasks.append(task_id)
        self.in_progress_tasks.remove(task_id)

    def get_universal_task_number(self):
        return sum(1 for task in self.task_sequence if task.universal)

    def change_agent(self, task_id, new_agent_name):
        # Find the task with the given task_id
        matching_task = next((task for task in self.task_sequence if task.id == task_id), None)

        if matching_task:
            matching_task.agent = [new_agent_name]

    def validate(self):
        valid = True
        messages: list[str] = []
        for task in self.task_sequence:
            start: int = task.finish[0] - task.finish[2] - task.finish[3]
            ends: list[int] = []
            for dep in task.conditions:
                ends.append(self.task_sequence[dep].finish[0] - self.task_sequence[dep].finish[3])

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


class Action(BaseModel):
    object: str
    place: str

class Task(BaseModel):
    """
    Represents a task to be completed.

    :param task_description: Dictionary containing task details.
    :type task_description: dict
    """
    id: int
    action: Action
    state: TaskState | None = None
    conditions: list[int] 
    universal: bool
    agent: list[str]
    start: int | None = None
    finish: int | None = None

    distribution: list[tuple[tuple[int, int], tuple[int, int], tuple[float, float]]]
    rejection_prob: float


    # def __init__(self, task_description):
    #     self.id: int = task_description['ID']
    #     self.action: dict[str, str] = {'Object': task_description['Object'], 'Place': task_description['Place']}
    #     self.state: TaskState |None = None
    #     self.conditions: list[int] = task_description['Conditions']
    #     self.universal: bool = task_description['Agent'] == 'Both'
    #     self.agent: str = task_description['Agent']
    #     self.start: int | None = None
    #     self.finish: int | None = None

    #     self.distribution: list[list[np.ndarray]] = task_description["Distribution"]
    #     self.rejection_prob: float = task_description["Rejection_prob"]

    def __str__(self):
        """
        Returns string representation of task.

        :return: String representation of task.
        :rtype: str
        """
        try:
            s1 = f"ID: {self.id}, agent: {self.agent[0]}, status: {TaskState(self.state).name}, " + \
                f"task action: {self.action}, conditions: {self.conditions}, universal: {self.universal}, " + \
                f"start: {self.start}, finish: {self.finish}"
            logging.info(s1)
        except ValueError:
            s1 = f"ID: {self.id}, agent: {self.agent[0]}, state: {self.state}, " + \
                    f"task action: {self.action}, conditions: {self.conditions}, universal: {self.universal}, " + \
                    f"start: {self.start}, finish: {self.finish}"
            logging.info(s1)
        return s1
    
    def __repr__(self) -> str:
        return self.__str__()

    def progress(self, current_time: int, duration:int) -> float:
        """
        Calculates and returns the progress of the task as a percentage.

        :param current_time: Current time.
        :type current_time: int
        :param duration: Total duration of task.
        :type duration: int
        :return: Progress of task as a percentage.
        :rtype: float
        """
        if self.start is None:
            return 0.0
        return round(number=(current_time - self.start / float(duration)) * 100, ndigits=2)

    def as_dict(self):
        """
        Returns task details as a dictionary.

        :return: Task details as a dictionary.
        :rtype: dict
        """
        state = TaskState(self.state).name if self.state is not None else None
        return {
            'agent': copy.deepcopy(self.agent),
            'id': copy.deepcopy(self.id),
            'action': copy.deepcopy(self.action),
            'status': copy.deepcopy(state),
            'conditions': copy.deepcopy(self.conditions),
            'universal': copy.deepcopy(self.universal),
            'start': copy.deepcopy(self.start),
            'finish': copy.deepcopy(self.finish)}

    def get_duration(self, rand_gen: Optional[np.random.Generator] = None, seed=None, distribution_param=None):
        if rand_gen is None:
            rand_gen = np.random.default_rng(seed + self.id)
        assert isinstance(rand_gen, np.random.Generator)

        if distribution_param is None:
            distribution_param = self.distribution

        duration = []
        for phase in range(3):
            duration.append(self.get_phase_duration(phase, distribution_param, rand_gen, seed))

        return [sum(duration), duration[0], duration[1], duration[2]]

    def get_phase_distribution(self, phase, distribution_param, rand_gen: Optional[np.random.Generator]=None,  seed=None):
        if rand_gen is None:
            rand_gen = np.random.default_rng(seed + self.id)

        distributions = []
        # Create distribution for each set of parameters
        for mean, scale, fail_prob in zip(*distribution_param[phase]):
            distributions.append(rand_gen.normal(loc=mean,
                                                 scale=scale,
                                                 size=int(fail_prob * 1000)))
        return np.concatenate(distributions)

    def get_phase_duration(self, phase, distribution_param, rand_gen: Optional[np.random.Generator] = None,  seed=None):
        if rand_gen is None:
            rand_gen = np.random.default_rng(seed + self.id)
        assert isinstance(rand_gen, np.random.Generator)

        distribution = self.get_phase_distribution(phase, distribution_param, rand_gen, seed)

        # Choice a random value from the concatenated distribution
        sample = int(np.round(rand_gen.choice(distribution), decimals=0))
        if sample <= 0:
            sample = 1

        return sample
