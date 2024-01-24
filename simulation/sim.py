"""
    Simulation class probability simulation

    @author: Marina Ionova, student of Cybernetics and Robotics at the CTU in Prague
    @contact: marina.ionova@cvut.cz
"""
from simulation.task_execution_time_const import get_approximated_task_duration
from numpy.random import Generator, choice
import numpy as np
import logging
import json
import time


class Sim:
    """
    A class that simulates the execution of tasks based on the probability
    distribution of their duration, as well as the choice of a person
    who is offered to him by the control logic.
    """
    def __init__(self, agent_name, job, **kwargs):
        self.agent_name = agent_name
        self.job = job
        self.task_duration = {}
        self.weights = None
        self.prob = None
        self.seed = None
        self.fail_probability = []
        self.task_execution = {'Human': {'Start': 0, 'Duration': []}, 'Robot': {'Start': 0, 'Duration': []}}
        self.start_time = time.time()

        # self.set_param()
        self.set_tasks_duration(**kwargs)

    def set_param(self):
        """
        Sets simulation parameters from config file.
        """
        with open('./simulation/config.json') as f:
            param = json.load(f)
        self.seed = param['Seed']
        self.weights = param['Allocation weights']
        self.fail_probability = param['Fail probability']

    def set_tasks_duration(self, **kwargs):
        for task in self.job.task_sequence:
            self.task_duration[task.id] = set_task_time(task, self.agent_name, **kwargs)

    def set_task_end(self, agent, job, current_time):
        """
        Setting the end time of a task for a given agent in a job.  If there is a dependent task
        that overlaps with the current task, it adjusts the duration to account for the overlapping time.

        :param agent: Agent assigned to the task
        :type agent: Agent
        :param job: Job to which task belongs.
        :type job: Job
        :param current_time: Current simulation time.
        :type current_time: int
        :return: task completion time
        :rtype: int
        """
        self.task_execution[agent.name]['Start'] = current_time
        self.task_execution[agent.name]['Duration'] = self.task_duration[agent.current_task.id]
        dependent_task = check_dependencies(job, agent.current_task)
        if dependent_task:
            overlapping = dependent_task.start + sum(
                self.task_execution[dependent_task.agent]['Duration'][1:3]) - current_time
            if overlapping > self.task_execution[agent.name]['Duration'][1]:
                self.task_execution[agent.name]['Duration'][0] += overlapping - \
                                                                  self.task_execution[agent.name]['Duration'][1]

        return current_time + self.task_execution[agent.name]['Duration'][0]

    def ask_human(self, question_type, task):
        """
        Simulates a person's choice of accepting or rejecting a
        proposal to complete a task or change a schedule.

        :param question_type: Type of question to ask agent.
        :type question_type: str
        :param task: Task to ask agent about.
        :type task: Task
        :return: Agent's response to question.
        :rtype: bool
        """
        nameList = [True, False]
        if question_type == 'change_agent':
            if task.agent == 'Robot':
                np.random.seed(self.seed + task.id)
                answer = choice(nameList,  p=(task.get_reject_prob(), 1-task.get_reject_prob()), size=1)
                logging.info(f'Offer to complete task {task.id} instead of robot. Answer {answer[0]}')
                return answer[0]
            else:
                np.random.seed(self.seed + task.id)
                answer = choice(nameList, p=(1-task.get_reject_prob(), task.get_reject_prob()), size=1)
                logging.info(f'Offer to complete task {task.id} instead of human. Answer {answer[0]}')
                return answer[0]
        elif question_type == 'execute_task':
            np.random.seed(self.seed + task.id)
            answer = choice(nameList, p=(1 - task.get_reject_prob(), task.get_reject_prob()), size=1)
            logging.info(f'Offer to complete task {task.id}. Answer {answer[0]}')
            return answer[0]
        return False

    def get_feedback_from_robot(self, task, job, current_time):
        """
        Checks the status of a task being executed by a robot agent.

        :param task: Task being executed.
        :type task: Task
        :param job: Job containing task.
        :type job: Job
        :param current_time: Current time in simulation.
        :type current_time: int
        :return: Status of task and time information.
        :rtype: tuple
        """
        if self.task_execution['Robot']['Duration'][0] != 0:
            logging.debug(
                f'start: {self.task_execution["Robot"]["Start"]}, duration :{self.task_execution["Robot"]["Duration"]}')
            if current_time < (self.task_execution['Robot']['Start'] + self.task_execution['Robot']['Duration'][1]):
                return 'Preparation', -1
            elif current_time < (self.task_execution['Robot']['Start'] + self.task_execution['Robot']['Duration'][0] -
                                 self.task_execution['Robot']['Duration'][3]):
                dependent_task = check_dependencies(job, task)
                if dependent_task and current_time < (self.task_execution['Human']['Start'] +
                                                      (self.task_execution['Human']['Duration'][0] -
                                                       self.task_execution['Human']['Duration'][3])):
                    return 'Waiting', current_time - (self.task_execution['Robot']['Start'] +
                                                      self.task_execution['Robot']['Duration'][1])
                else:
                    return 'Execution', -1

            elif current_time < (self.task_execution['Robot']['Start'] + self.task_execution['Robot']['Duration'][0]):
                return 'Completion', -1
            else:
                time_info = self.task_execution['Robot']['Duration']
                time_info[0] += self.task_execution['Robot']['Start']
                self.task_execution['Robot']['Start'] = 0
                self.task_execution['Robot']['Duration'] = [0, 0, 0, 0]
                return 'Completed', time_info

    def check_human_task(self, task, job, current_time):
        """
        Checks the status of a task being executed by a human agent.

        :param task: Task being executed.
        :type task: Task
        :param job: Job containing task.
        :type job: Job
        :param current_time: Current time in simulation.
        :type current_time: int
        :return: Status of task and time information.
        :rtype: tuple
        """
        if self.task_execution['Human']['Duration'] != 0:
            if (self.task_execution['Human']['Start'] + self.task_execution['Human']['Duration'][0]) \
                    > current_time:
                dependent_task = check_dependencies(job, task)
                if dependent_task and current_time < (self.task_execution['Robot']['Start'] +
                                                      (self.task_execution['Robot']['Duration'][0] -
                                                       self.task_execution['Robot']['Duration'][3])):
                    return 'Waiting', current_time - (self.task_execution['Human']['Start'] +
                                                      self.task_execution['Human']['Duration'][1])
                else:
                    return 'In progress', -1
            else:
                time_info = self.task_execution['Human']['Duration']
                time_info[0] += self.task_execution['Human']['Start']
                self.task_execution['Human']['Start'] = 0
                self.task_execution['Human']['Duration'] = [0, 0, 0, 0]
                return 'Completed', time_info


# def get_param(param_name):
#     with open('./simulation/config.json', 'r') as f:
#         sim_param = json.load(f)
#     return sim_param[param_name]


def set_task_time(task, agent=None, **kwargs):
    if isinstance(task, dict):
        if not agent:
            agent = task['Agent']
        action = task['Action']
        ID = task['ID']
    else:
        if not agent:
            agent = task.agent
        action = task.action
        ID = task.id
    duration = get_approximated_task_duration(agent, action)
    logging.debug(f'{agent}, {action}, {[sum(duration), duration[0], duration[1], duration[2]]}')
    if duration[0] != 0:
        # scale = 2 if agent == 'Human' else 1
        for i in range(len(duration)):
            # Generate samples from the first Gaussian component
            np.random.seed(kwargs['seed'] + ID)
            samples1 = np.random.normal(loc=duration[i],
                                        scale=kwargs['scale'],
                                        size=int(kwargs['fail_prob'][1] * 1000))
            # Generate samples from the second Gaussian component
            np.random.seed(kwargs['seed'] + ID)
            samples2 = np.random.normal(loc=duration[i] * kwargs['second_mode'][0],
                                        scale=kwargs['scale'] * kwargs['second_mode'][1],
                                        size=int(kwargs['fail_prob'][0] * 1000))
            # Concatenate the samples from both components
            samples = np.concatenate((samples1, samples2))
            # Choice a random value from the concatenated samples
            np.random.seed(kwargs['seed'] + ID)
            sample = int(np.random.choice(samples))
            if sample <= 0:
                sample = 1
            duration[i] = sample

    logging.debug(f'{agent}, {action}, {[sum(duration), duration[0], duration[1], duration[2]]}')
    return [sum(duration), duration[0], duration[1], duration[2]]


def check_dependencies(job, task):
    for another_task in job.task_sequence:
        if (another_task.id in task.conditions) & (another_task.status == 1):
            return another_task
    return None