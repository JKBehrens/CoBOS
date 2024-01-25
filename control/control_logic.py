"""
    ControlLogic class for job execution based on reactive schedule

    @author: Marina Ionova, student of Cybernetics and Robotics at the CTU in Prague
    @contact: marina.ionova@cvut.cz
"""
import threading

import pandas as pd

from visualization import Vis, initial_and_final_schedule, Web_vis
from scheduling import Schedule, print_schedule
from control.agents import Agent
from control.jobs import Job
import logging
import json
import time


class ControlLogic:
    """
    Class that controls the scheduling and execution of tasks by agents.

    :param case: Case to be executed.
    :type case: str
    """
    def __init__(self, case):
        self.case = case
        self.agent_list = ['Robot', 'Human']
        self.agents = []
        self.current_time = 0
        self.start_time = time.time()
        self.task_finish_time = []
        self.schedule_model = None
        self.available_tasks = []
        self.FAIL = False

        self.job = Job(self.case)
        self.set_schedule()

        # self.plot = Vis(horizon=self.schedule_model.horizon)
        self.plot = None

    def set_schedule(self):
        """
        Sets the schedule for task execution by agents.
        """
        self.schedule_model = Schedule(self.job)
        schedule = self.schedule_model.set_schedule(seed=10, fail_prob=[0.1, 0.9], second_mode=[3, 3], scale=2)
        if not schedule:
            self.FAIL = True
            logging.error('Scheduling failed')
        else:
            for agent_name in self.agent_list:
                self.agents.append(Agent(agent_name, schedule[agent_name], self.job,
                                 seed=7, fail_prob=[0.1, 0.9], second_mode=[3, 3], scale=2))

            self.job.predicted_makespan = self.job.get_current_makespan()
        self.set_task_status()

    def set_task_status(self):
        """
        Sets the status of tasks for each agent.
        """
        for agent in self.agents:
            for i, task in enumerate(agent.tasks):
                task.status = -1 if len(task.conditions) != 0 else 0
            agent.refresh_task_availability()

    def find_coworker_task(self, agent):
        """
        If the agent has run out of available tasks in his list, he looks for
        universal tasks in the list of a colleague that he can perform instead
        of him to speed up the process.

        :param agent: Agent to find coworker task for.
        :type agent: Agent
        :return: True if coworker task is found and executed, False otherwise.
        :rtype: bool
        """
        coworker = self.agents[self.agents.index(agent) - 1]
        updated_available_tasks = coworker.get_available_universal_tasks()
        if updated_available_tasks is not None and updated_available_tasks != self.available_tasks:
            self.available_tasks = updated_available_tasks
            # rescheduling estimation
            self.schedule_model.refresh_variables(self.current_time)
            makespan_and_task = self.schedule_model.set_list_of_possible_changes(self.available_tasks, agent)
            if makespan_and_task and makespan_and_task[0][0] < self.job.get_current_makespan():
                for coworker_task in makespan_and_task:
                    if (agent.name == 'Human' and coworker_task[1].id not in agent.rejection_tasks) \
                            or agent.name == 'Robot':
                        if agent.ask_human('change_agent', coworker_task[1]):
                            self.change_agent(coworker_task[1], coworker)
                            agent.execute_task(coworker_task[1], self.job, self.current_time)
                            self.update_tasks_status()
                            if self.plot:
                                try:
                                    self.plot.update_info(agent, start=True)
                                except AttributeError:
                                    pass
                            return True
                        else:
                            agent.rejection_tasks.append(coworker_task[1].id)
        return False

    def change_agent(self, task, current_agent):
        """
        Changes the agent assigned to a task.

        :param task: Task to change agent for.
        :type task: Task
        :param current_agent: Current agent assigned to the task.
        :type current_agent: Agent
        """
        task.agent = self.agents[self.agents.index(current_agent) - 1].name
        self.schedule_model.set_new_agent(task)
        self.schedule_model.refresh_variables(self.current_time)
        schedule = self.schedule_model.solve()
        for agent in self.agents:
            agent.refresh_tasks(schedule[agent.name])
        logging.info('____RESCHEDULING______')
        print_schedule(schedule)
        logging.info('______________________')

    def update_tasks_status(self):
        """
        Updates the status of tasks based on their dependencies.
        """
        for agent in self.agents:
            for i, task in enumerate(agent.tasks):
                if len(task.conditions) != 0 and task.status == -1:
                    if set(task.conditions).issubset(self.job.completed_tasks + self.job.in_progress_tasks):
                        task.status = 0
                        agent.refresh_task_availability()

    def check_task_progress(self):
        """
        Checks the progress of each agent's current task and updates the schedule accordingly.
        """
        for agent in self.agents:
            if not agent.availability:
                coworker = self.agents[self.agents.index(agent) - 1]
                status, time_info = agent.get_feedback(self.job, self.current_time, coworker)
                logging.debug(f'Status{status}')
                if status == 'Completed':
                    self.task_completed(agent, time_info)
                elif status == 'Waiting':
                    agent.waiting = time_info

    def shift_schedule(self):
        """
       Shifts the schedule forward by one time unit if a task has been completed.
       """
        for agent in self.agents:
            shift = False
            for task in agent.tasks:
                if task.status == 1 and task.finish[0] < self.current_time:
                    task.finish[0] = self.current_time
                    shift = True
                elif shift and (task.status == -1 or task.status == 0):
                    task.start += 1
                    task.finish[0] += 1

    def task_completed(self, agent, time_info):
        """
        Updates the status of a completed task and logs the completion.
        """
        agent.finish_task(time_info)
        self.job.refresh_completed_task_list(agent.current_task.id)
        logging.info(
            f'TIME {self.current_time}. {agent.name} completed the task {agent.current_task.id}. Progress {self.job.progress()}.')
        # if self.plot:
        #     self.plot.update_info(agent)

    def schedule_as_dict(self, hierarchy=False):
        """
        Returns the current schedule as a dictionary.
        """
        if hierarchy:
            output = {}
            for agent in self.agents:
                output[agent.name] = agent.tasks_as_dict()
        else:
            output = {
                "Status": [],
                "Start": [],
                "End": [],
                "Agent": [],
                "ID": [],
                "Conditions": [],
                "Object": [],
                "Place": []
            }
            for agent in self.agents:
                for task in agent.tasks_as_dict():
                    output['Status'].append(task['Status'])
                    output['Start'].append(task['Start'])
                    output['End'].append(task['Finish'][0])
                    output['ID'].append(task['ID'])
                    output['Conditions'].append(task['Conditions'])
                    output['Object'].append(task['Action']['Object'])
                    output['Place'].append(task['Action']['Place'])

                    if task['Universal']:
                        output['Agent'].append(f'Assigned\n to {task["Agent"]}')
                    else:
                        output['Agent'].append(task['Agent'])
        return output

    def run(self, animation=False, online_plot=False, experiments=False):
        """
        Run the scheduling simulation.
        """
        schedule_data = [self.schedule_as_dict(hierarchy=True)]

        if animation:
            self.plot = Vis(horizon=self.schedule_model.horizon)
            self.plot.delete_existing_file()
        if online_plot:
            self.plot = Web_vis(data=self.schedule_as_dict())

        while True:
            if self.job.progress() == 100:
                break
            self.check_task_progress()
            for agent in self.agents:
                logging.debug(f'TIME: {self.current_time}. Is {agent.name} available? {agent.availability}')
                if agent.availability:
                    task = agent.find_your_task(self)
                    if task is None:
                        self.find_coworker_task(agent)
                    else:
                        agent.execute_task(task, self.job, self.current_time)
                        self.update_tasks_status()
                        if online_plot:
                            self.plot.update_info(agent, start=True)

            self.current_time += 1
            self.shift_schedule()

            if online_plot:
                self.plot.current_time = self.current_time
                self.plot.data = self.schedule_as_dict()
                self.plot.update_gantt_chart()
                self.plot.update_dependency_graph()
                time.sleep(1)

            if animation:
                # save current state
                if self.plot.current_time + 2 == self.current_time:
                    self.plot.current_time = self.current_time
                    self.plot.data = self.schedule_as_dict(hierarchy=True)
                    self.plot.save_data()

        logging.info('__________FINAL SCHEDULE___________')
        for agent in self.agents:
            logging.info(agent.name)
            agent.print_tasks()
        logging.info('___________________________________')
        logging.info(f'SIMULATION TOTAL TIME: {time.time() - self.start_time}')
        schedule_data.append(self.schedule_as_dict(hierarchy=True))
        with open(initial_and_final_schedule, 'w') as f:
            json.dump(schedule_data, f, indent=4)

        if experiments:
            statistics = {}
            statistics['makespan'] = [self.job.predicted_makespan, self.job.get_current_makespan()]
            statistics['rejection number'] = len(self.agents[1].rejection_tasks)
            statistics['runtime'] = [self.schedule_model.rescheduling_run_time, self.schedule_model.evaluation_run_time]
            return schedule_data, statistics

