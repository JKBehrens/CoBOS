"""
    Agent class for control and communication with agents

    @author: Marina Ionova, student of Cybernetics and Robotics at the CTU in Prague
    @contact: marina.ionova@cvut.cz
"""
from simulation.sim import Sim
import logging


class Agent(Sim):
    """
    Represents an agent in the simulation.

    :param name: Name of the agent.
    :type name: str
    :param tasks: List of tasks assigned to the agent.
    :type tasks: list
    """
    def __init__(self, name, tasks=None, job=None, **kwargs):
        super().__init__(name, job, **kwargs)
        self.name = name
        self.tasks = tasks
        self.availability = True
        self.current_task = None
        self.available_tasks = []
        self.rejection_tasks = []
        self.delay = 0
        self.waiting = 0

    def set_start_task(self, task, start):
        """
        Sets the start time of the current task.

        :param task: Task to be started.
        :type task: Task
        :param start: Start time of the task.
        :type start: int
        """
        self.availability = False
        self.current_task = task
        task.start = start
        task.status = 1
        self.available_tasks.remove(task)

    def finish_task(self, time_info):
        """
        Finishes the current task.

        :param time_info: Time information of the task's phases.
        :type time_info: list
        """
        self.current_task.finish = time_info
        self.current_task.status = 2
        self.waiting = 0
        self.availability = True

    def print_tasks(self):
        """
        Prints the list of tasks assigned to the agent.
        """
        for task in self.tasks:
            task.__str__()

    def print_current_state(self):
        """
        Returns the current state of the agent.

        :return: Current state of the agent.
        :rtype: str
        """
        if self.availability:
            return 'waiting for task'
        else:
            return f'is doing {self.current_task.action}'

    def refresh_tasks(self, tasks):
        """
        Refreshes the list of tasks assigned to the agent.

        :param tasks: List of tasks assigned to the agent.
        :type tasks: list
        """
        self.tasks = tasks
        self.refresh_task_availability()

    def refresh_task_availability(self):
        """
        Refreshes the list of available tasks.
        """
        self.available_tasks = []
        for task in self.tasks:
            if task.status == 0:
                self.available_tasks.append(task)

    def get_current_task_idx(self):
        """
        Returns the index of the current task.

        :return: Index of the current task.
        :rtype: int
        """
        for i, task in enumerate(self.tasks):
            if task.status == 1 or task.status == 0:
                return i
            elif task.status == -1:
                return i - 1
        return len(self.tasks)

    def get_available_universal_tasks(self):
        """
        Returns the list of available universal tasks.

        :return: List of available universal tasks.
        :rtype: list
        """
        universal_tasks = [task for task in self.available_tasks if task.universal]
        if len(universal_tasks) == 0:
            return None
        return universal_tasks

    def remove_task(self, task):
        """
        Removes a task from the list of tasks assigned to the agent.

        :param task: Task to be removed.
        :type task: Task
        """
        self.tasks.remove(task)
        if task in self.available_tasks:
            self.available_tasks.remove(task)

    def add_task(self, task):
        """
        Adds a task to the list of tasks assigned to the agent.

        :param task: Task to be added.
        :type task: Task
        """
        idx = self.get_current_task_idx()
        self.tasks.insert(idx + 1, task)
        task.agent = self.name

    def tasks_as_dict(self):
        """
        Returns the list of tasks assigned to the agent as a dictionary.

        :return: List of tasks assigned to the agent as a dictionary.
        :rtype: list
        """
        output = []
        for task in self.tasks:
            output.append(task.as_dict())
        return output

    def find_your_task(self, cl):
        """
        Finds the next available task for the agent.

        :param cl: ControlLogic class.
        :type cl: ControlLogic
        :return: Assigned task
        :rtype: Task
        """
        if not self.availability:
            self.current_task.status = -1
            self.refresh_task_availability()
        for i, task in enumerate(self.available_tasks):
            if task.universal and self.name == 'Human':
                if self.ask_human('execute_task', task):
                    logging.info(f'Human has agreed. Task in progress...')
                    return task
                else:
                    logging.info(f'Human has not agreed to the task.')
                    self.rejection_tasks.append(task.id)
                    cl.change_agent(task=task, current_agent=self)
            else:
                return task

        if not self.availability:
            self.current_task.status = 0
            self.refresh_task_availability()
        return None

    def execute_task(self, task, job, current_time, coworker):
        """
        Executes a task and logs the action.

        :param task: Task to be executed.
        :type task: Task
        :param job: Job to which task belongs.
        :type job: Job
        :param current_time: Current time.
        :type current_time: int
        """
        self.task_execution[coworker.name] = coworker.task_execution[coworker.name]
        self.set_start_task(task, current_time)
        self.set_task_end(self, job, current_time)
        job.in_progress_tasks.append(task.id)
        logging.info(f'{task.agent} is doing the task {task.id}. Place object {task.action["Object"]}'
                     f'to {task.action["Place"]}. TIME {current_time}')

    def get_feedback(self, job, current_time, coworker):
        """
        Sends feedback from an agent.

        :param job: Job to which task belongs.
        :type job: Job
        :param current_time: Current time.
        :type current_time: int
        :param coworker: Coworker
        :type coworker: Agent
        :return: Feedback from agent.
        :rtype: str
        """
        self.task_execution[coworker.name] = coworker.task_execution[coworker.name]
        if self.name == 'Robot':
            return self.get_feedback_from_robot(self.current_task, job, current_time)
        else:
            return self.check_human_task(self.current_task, job, current_time)
