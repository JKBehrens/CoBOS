"""
    Simulation class probability simulation

    @author: Marina Ionova, student of Cybernetics and Robotics at the CTU in Prague
    @contact: marina.ionova@cvut.cz
"""
from typing import Optional
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
        self.task_duration = {"Human": {}, "Robot": {}}
        self.prob = None
        if "seed" in kwargs:
            self.seed = int(kwargs["seed"])
        self.rand = np.random.default_rng(seed=self.seed)
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
        self.task_duration = self.job.task_duration(rand_gen=self.rand)


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
        self.task_execution[agent.name]['Duration'] = self.task_duration[agent.name][agent.current_task.id]
        dependent_task = job.get_in_process_dependent_task(agent.current_task)
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
                answer = choice(nameList,  p=(task.rejection_prob, 1-task.rejection_prob), size=1)
                logging.info(f'Offer to complete task {task.id} instead of robot. Answer {answer[0]}')
                return answer[0]
            else:
                np.random.seed(self.seed + task.id)
                answer = choice(nameList, p=(1-task.rejection_prob, task.rejection_prob), size=1)
                logging.info(f'Offer to complete task {task.id} instead of human. Answer {answer[0]}')
                return answer[0]
        elif question_type == 'execute_task':
            np.random.seed(self.seed + task.id)
            answer = choice(nameList, p=(1 - task.rejection_prob, task.rejection_prob), size=1)
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
                f'Robot: start: {self.task_execution["Robot"]["Start"]}, duration :{self.task_execution["Robot"]["Duration"]}')
            if current_time < (self.task_execution['Robot']['Start'] + self.task_execution['Robot']['Duration'][1]):
                return 'Preparation', -1
            elif current_time < (self.task_execution['Robot']['Start'] + self.task_execution['Robot']['Duration'][0] -
                                 self.task_execution['Robot']['Duration'][3]):
                dependent_task = job.get_in_process_dependent_task(task)

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
            logging.debug(
                f'Human: start: {self.task_execution["Human"]["Start"]}, duration :{self.task_execution["Human"]["Duration"]}')

            if (self.task_execution['Human']['Start'] + self.task_execution['Human']['Duration'][0]) \
                    > current_time:
                dependent_task = job.get_in_process_dependent_task(task)
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
