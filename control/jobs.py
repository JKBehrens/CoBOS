"""
    Job class contains the variables to describe the job and the functions needed to define it.
    Task class contains the variables needed to describe the task.

    @author: Marina Ionova, student of Cybernetics and Robotics at the CTU in Prague
    @contact: marina.ionova@cvut.cz
"""
from inputs import case_generator
from typing import Optional
import numpy as np
import logging
import copy


class Job:
    """
    A class representing a job consisting of multiple tasks.

    :param case: Input case for generating job description.
    :type case: str
    """

    def __init__(self, case, seed):
        self.case = str(case)
        self.job_description = case_generator.set_input(self.case, seed)
        self.task_sequence = [Task(task) for task in self.job_description]
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
        for task in self.task_sequence:
            task.__str__()
        logging.info("____________________________")

    def progress(self):
        """
        Returns the percentage of completed tasks in the job.
        """
        completed_tasks = sum( 1 for task in self.task_sequence if task.status == 2)
        return round((completed_tasks / self.task_number) * 100, 2)

    def get_current_makespan(self):
        """
        Returns the current makespan of the job.
        """
        return max(task.finish[0] for task in self.task_sequence)

    def get_completed_and_in_progress_task_list(self):
        output = []
        for task in self.task_sequence:
            if task.status == 2 or task.status == 1:
                output.append(task.id)
        return output

    def task_duration(self,  rand_gen: Optional[np.random.Generator] = None,  seed=None):
        if rand_gen is None:
            rand_gen = np.random.default_rng(seed)
        assert isinstance(rand_gen, np.random.Generator)

        task_duration = {"Human": {}, "Robot": {}}
        for task in self.task_sequence:
            task_duration["Human"][task.id] = task.get_duration(rand_gen=rand_gen)
            task_duration["Robot"][task.id] = task.get_duration(rand_gen=rand_gen)

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

    def get_in_process_dependent_task(self, task):
        for other_task_id in task.conditions:
            another_task = self.task_sequence[other_task_id]
            assert other_task_id == another_task.id
            if another_task.status == 1:
                return another_task
        return None

    def get_universal_task_number(self):
        return sum(1 for task in self.task_sequence if task.universal)

    def change_agent(self, task_id, new_agent_name):
        # Find the task with the given task_id
        matching_task = next((task for task in self.task_sequence if task.id == task_id), None)

        if matching_task:
            matching_task.agent = new_agent_name

class Task:
    """
    Represents a task to be completed.

    :param task_description: Dictionary containing task details.
    :type task_description: dict
    """

    def __init__(self, task_description):
        self.id = task_description['ID']
        self.action = {'Object': task_description['Object'], 'Place': task_description['Place']}
        self.status = None
        self.conditions = task_description['Conditions']
        self.universal = task_description['Agent'] == 'Both'
        self.agent = task_description['Agent']
        self.start = None
        self.finish = None

        self.distribution = task_description["Distribution"]
        self.rejection_prob = task_description["Rejection_prob"]

    def __str__(self):
        """
        Returns string representation of task.

        :return: String representation of task.
        :rtype: str
        """
        logging.info(f"ID: {self.id}, agent: {self.agent}, status: {self.status}, "
                     f"task action: {self.action}, conditions: {self.conditions}, universal: {self.universal}, "
                     f"start: {self.start}, finish: {self.finish}")

    def progress(self, current_time, duration):
        """
        Calculates and returns the progress of the task as a percentage.

        :param current_time: Current time.
        :type current_time: int
        :param duration: Total duration of task.
        :type duration: int
        :return: Progress of task as a percentage.
        :rtype: float
        """
        return round((current_time - self.start / float(duration)) * 100, 2)

    def as_dict(self):
        """
        Returns task details as a dictionary.

        :return: Task details as a dictionary.
        :rtype: dict
        """
        return {
            'Agent': copy.deepcopy(self.agent),
            'ID': copy.deepcopy(self.id),
            'Action': copy.deepcopy(self.action),
            'Status': copy.deepcopy(self.status),
            'Conditions': copy.deepcopy(self.conditions),
            'Universal': copy.deepcopy(self.universal),
            'Start': copy.deepcopy(self.start),
            'Finish': copy.deepcopy(self.finish)}

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
