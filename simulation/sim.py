"""
    Simulation class probability simulation

    @author: Marina Ionova, student of Cybernetics and Robotics at the CTU in Prague
    @contact: marina.ionova@cvut.cz
"""
import copy
from typing import Optional
from control.jobs import Job
from simulation.task_execution_time_const import get_approximated_task_duration
from control.agent_and_task_states import AgentState
from numpy.random import Generator
from inputs.config import response_time_max, response_time_min
import numpy as np
import logging
import json
import time

SIM_MODE = 'NORMAL'  # OVERFIT, UNDERFIT, NORMAL


class Sim:
    """
    A class that simulates the execution of tasks based on the probability
    distribution of their duration, as well as the choice of a person
    who is offered to him by the control logic.
    """
    def __init__(self, agent_name, job, seed, **kwargs):
        self.agent_name = agent_name
        self.job: Job = copy.deepcopy(job)
        self.agent_list = ['Human', 'Robot']
        self.task_duration = {agent: {} for agent in self.agent_list}
        self.prob = None
        self.seed = seed
        self.rand = np.random.default_rng(seed=self.seed)
        self.fail_probability = []
        self.task_execution = {agent: {'Start': 0, 'Duration': []} for agent in self.agent_list}
        self.start_time = time.time()

        self.set_tasks_duration(**kwargs)

        answer_seed = kwargs.get('answer_seed', None)
        self.human_answer = self.sample_human_response(answer_seed)
        self.task_acceptance: dict[int, bool] = dict(
            zip(
                self.human_answer["execute_task"].keys(),
                [v[0] for v in self.human_answer["execute_task"].values()],
            )
        )


        self.response_time = []
        self.set_response_time()

    def _get_deterministic_job(self) -> Job:
        """returns a deterministic version of the job. All universal tasks that are rejected 
        will be converted to tasks for the other agent(s).
        Task durations are fixed.

        Returns:
            Job: Job with only true universal tasks
        """
        out_job: Job = copy.deepcopy(self.job)
        for task in out_job.task_sequence:
            if task.universal and self.task_acceptance[task.id]:
                task.rejection_prob = 0.0
            elif task.universal and not self.task_acceptance[task.id]:
                task.agent = self.agent_list.copy()
                task.agent.remove(self.agent_name)
                task.universal = False
                task.rejection_prob = 0.0
            else:
                # print(task)
                task.rejection_prob = 0.0

        return out_job

    def set_tasks_duration(self, **kwargs):
        for task in self.job.task_sequence:
            if SIM_MODE == 'NORMAL':
                self.task_duration["Robot"][task.id] = task.get_duration(rand_gen=self.rand)
                self.task_duration["Human"][task.id] = task.get_duration(rand_gen=self.rand)
            elif SIM_MODE == 'OVERFIT':
                new_distribution_param = []
                for phase_param in task.distribution:
                    mean = [phase_param[0][0] + 2, phase_param[0][1] + 2]
                    new_distribution_param.append([mean, phase_param[1], phase_param[2]])
                self.task_duration["Human"][task.id] = task.get_duration(rand_gen=self.rand,
                                                                         distribution_param=new_distribution_param)
                self.task_duration["Robot"][task.id] = task.get_duration(rand_gen=self.rand,
                                                                         distribution_param=new_distribution_param)
            elif SIM_MODE == 'UNDERFIT':
                new_distribution_param = []
                for phase_param in task.distribution:
                    mean = [phase_param[0][0] - 2 if phase_param[0][0] > 2 else 1, phase_param[0][1] - 2]
                    new_distribution_param.append([mean, phase_param[1], phase_param[2]])
                self.task_duration["Human"][task.id] = task.get_duration(rand_gen=self.rand,
                                                                         distribution_param=new_distribution_param)
                self.task_duration["Robot"][task.id] = task.get_duration(rand_gen=self.rand,
                                                                         distribution_param=new_distribution_param)

    def set_response_time(self):
        n = self.job.get_universal_task_number()
        self.response_time = self.rand.integers(response_time_min, response_time_max, size=n)

    def set_task_end(self, agent, current_time):
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

        coworker_name = self.agent_list[self.agent_list.index(agent.name)-1]
        if self.task_execution[coworker_name]['Duration'] != []:
            if sum(self.task_execution[coworker_name]['Duration'][1:]) == self.task_execution[coworker_name]['Duration'][0]:
                overlapping = self.task_execution[coworker_name]['Start'] + sum(
                    self.task_execution[coworker_name]['Duration'][1:3]) - current_time
            else:
                overlapping = self.task_execution[coworker_name]['Start'] + \
                    self.task_execution[coworker_name]['Duration'][0] \
                    - self.task_execution[coworker_name]['Duration'][3] - current_time
            logging.debug(f'overlapping:{overlapping}, dependent_task.start :{self.task_execution[coworker_name]["Start"]},'
                  f'duration of dependent task: {self.task_execution["Human"]["Duration"]}, time: {current_time},'
                          f' sampled duration of task: {self.task_duration[agent.name][agent.current_task.id]}')
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
        return self.human_answer[question_type][task.id]

    def get_feedback_from_robot(self, task, current_time):
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
                f'Robot: start: {self.task_execution}') #["Robot"]["Start"]}, duration :{self.task_execution["Robot"]["Duration"]}')
            if current_time < (self.task_execution['Robot']['Start'] + self.task_execution['Robot']['Duration'][1]):
                return AgentState.PREPARATION, -1
            elif current_time < (self.task_execution['Robot']['Start'] + self.task_execution['Robot']['Duration'][0] -
                                 self.task_execution['Robot']['Duration'][3]):
                coworker_name = self.agent_list[self.agent_list.index(task.agent) - 1]

                if self.task_execution[coworker_name]['Duration'] != [0, 0, 0, 0] and \
                        current_time < (self.task_execution['Robot']['Start'] +
                                                      (self.task_execution['Robot']['Duration'][0] -
                                                       self.task_execution['Robot']['Duration'][2] -
                                                       self.task_execution['Robot']['Duration'][3])):
                    return AgentState.WAITING, current_time - (self.task_execution['Robot']['Start'] +
                                                      self.task_execution['Robot']['Duration'][1])
                else:
                    return AgentState.EXECUTION, -1

            elif current_time < (self.task_execution['Robot']['Start'] + self.task_execution['Robot']['Duration'][0]):
                return AgentState.COMPLETION, -1
            else:
                time_info = self.task_execution['Robot']['Duration']
                time_info[0] += self.task_execution['Robot']['Start']
                self.task_execution['Robot']['Start'] = 0
                self.task_execution['Robot']['Duration'] = [0, 0, 0, 0]
                return AgentState.DONE, time_info
        return AgentState.IDLE, -1

    def check_human_task(self, current_time):
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
                f'Human: start: {self.task_execution}') #["Human"]["Start"]}, duration :{self.task_execution["Human"]["Duration"]}')

            if (self.task_execution['Human']['Start'] + self.task_execution['Human']['Duration'][0]) \
                    > current_time:
                return AgentState.InPROGRESS, -1
            else:
                time_info = self.task_execution['Human']['Duration']
                time_info[0] += self.task_execution['Human']['Start']
                self.task_execution['Human']['Start'] = 0
                self.task_execution['Human']['Duration'] = [0, 0, 0, 0]
                return AgentState.DONE, time_info 
        else:
            raise NotImplementedError("is this case relevant? DO we ever go here?")

    def sample_human_response(self, seed=None):
        nameList = [True, False]
        answer = {'execute_task': {}}
        if seed:
            rand = np.random.default_rng(seed=seed)
            for task in self.job.task_sequence:
                answer['execute_task'][task.id] = rand.choice(nameList, p=(1 - task.rejection_prob, task.rejection_prob), size=1)
        else:
            nameList = [True, False]
            for task in self.job.task_sequence:
                answer['execute_task'][task.id] = self.rand.choice(nameList, p=(1 - task.rejection_prob, task.rejection_prob), size=1)
        return answer