"""
    Job class contains the variables to describe the job and the functions needed to define it.
    Task class contains the variables needed to describe the task.

    @author: Marina Ionova, student of Cybernetics and Robotics at the CTU in Prague
    @contact: marina.ionova@cvut.cz
"""
from inputs import case_generator
import logging


class Job:
    """
    A class representing a job consisting of multiple tasks.

    :param case: Input case for generating job description.
    :type case: str
    """
    def __init__(self, case):
        self.case = str(case)
        self.job_description = case_generator.set_input(self.case)
        self.task_sequence = [Task(task) for task in self.job_description]
        self.in_progress_tasks = []
        self.completed_tasks = []
        self.agents = ["Human", "Robot"]
        self.task_number = len(self.task_sequence)

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
        return round((len(self.completed_tasks) / self.task_number) * 100, 2)

    def get_current_makespan(self):
        """
        Returns the current makespan of the job.
        """
        return max(task.finish if isinstance(task.finish, int) else task.finish[0] for task in self.task_sequence)

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

    def get_reject_prob(self):
        """
       Returns the probability of task rejection based on task ID.

       :return: Probability of task rejection.
       :rtype: float
       """
        task_rejection_prob = {0: 0.1, 1: 0.2, 2: 0.1, 3: 0.8,
                               4: 0.2, 5: 0.2, 6: 0.2, 7: 0.1,
                               8: 0.2, 9: 0.8, 10: 0.2, 11: 0.2,
                               12: 0.1, 13: 0.2, 14: 0.1, 15: 0.2}
        return task_rejection_prob[self.id]

    def as_dict(self):
        """
        Returns task details as a dictionary.

        :return: Task details as a dictionary.
        :rtype: dict
        """
        return {
            'Agent': self.agent,
            'ID': self.id,
            'Action': self.action,
            'Status': self.status,
            'Conditions': self.conditions,
            'Universal': self.universal,
            'Start': self.start,
            'Finish': self.finish}
