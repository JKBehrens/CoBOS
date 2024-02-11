"""
    ControlLogic class for job execution based on reactive schedule

    @author: Marina Ionova, student of Cybernetics and Robotics at the CTU in Prague
    @contact: marina.ionova@cvut.cz
"""
from visualization import Vis, initial_and_final_schedule, Web_vis
from scheduling import Schedule
from control.agent_states import AgentState
from control.agents import Agent
from control.jobs import Job
import logging
import json
import time
import copy

class ControlLogic:
    """
    Class that controls the scheduling and execution of tasks by agents.

    :param case: Case to be executed.
    :type case: str
    """
    def __init__(self, case, **kwargs):
        self.case = case
        self.agent_list = ['Robot', 'Human']
        self.agents = []
        self.current_time = 0
        self.start_time = time.time()
        self.task_finish_time = []
        self.schedule_model = None
        self.available_tasks = []
        self.output_data = []

        self.distribution_seed = kwargs.get('distribution_seed', 0)
        self.sim_seed = kwargs.get('sim_seed', 0)
        self.schedule_seed = kwargs.get('schedule_seed', 0)

        self.job = Job(self.case, seed=self.distribution_seed)
        self.set_schedule()

        # self.plot = Vis(horizon=self.schedule_model.horizon)
        self.plot = None

    def set_schedule(self):
        """
        Sets the schedule for task execution by agents.
        """
        self.schedule_model = Schedule(self.job, seed=self.schedule_seed)
        schedule = self.schedule_model.set_schedule()
        if not schedule:
            logging.error(f'Scheduling failed. Case {self.case}. Distribution seed {self.distribution_seed}, sim seed {self.sim_seed}, '
                          f'schedule seed {self.schedule_seed}')
            self.job.__str__()
            exit()
        else:
            for agent_name in self.agent_list:
                self.agents.append(Agent(agent_name, self.job, seed=self.sim_seed))
            self.job.predicted_makespan = self.job.get_current_makespan()
        self.set_task_status()

    def set_task_status(self):
        """
        Sets the status of tasks for each agent.
        """
        for i, task in enumerate(self.job.task_sequence):
            task.status = -1 if len(task.conditions) != 0 else 0

    def set_observation_data(self):
        """
        Checks the progress of each agent's current task and updates the schedule accordingly.
        """
        output = []
        for agent in self.agents:
            pass
            if agent.state == AgentState.REJECT:
                pass
            elif agent.state == AgentState.DONE:
                agent.state = AgentState.IDLE
            elif not agent.state == AgentState.IDLE:
                coworker = self.agents[self.agents.index(agent) - 1]
                agent.get_feedback(self.job, self.current_time, coworker=coworker)

    def shift_schedule(self):
        """
       Shifts the schedule forward by one time unit if a task has been completed.
       """
        for task in self.job.task_sequence:
        # for agent in self.agents:
            shift = False
            # for task in agent.tasks:
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
        # agent.finish_task(time_info)
        self.job.refresh_completed_task_list(agent.current_task.id)
        logging.info(
            f'TIME {self.current_time}. {agent.name} completed the task {agent.current_task.id}. Progress {self.job.progress()}.')
        # if self.plot:
        #     self.plot.update_info(agent)

    def run(self, animation=False, online_plot=False, experiments=False):
        """
        Run the scheduling simulation.
        """
        self.output_data.append(self.schedule_as_dict(hierarchy=True))

        if animation:
            self.plot = Vis(horizon=self.schedule_model.horizon)
            self.plot.delete_existing_file()
        if online_plot:
            self.plot = Web_vis(data=self.schedule_as_dict())

        while True:
            if self.job.progress() == 100:
                break
            logging.debug(f'TIME: {self.current_time}. Progress {self.job.progress()}')

            self.set_observation_data()

            # ask planner to decide about next actions for robot and human
            selected_task = self.schedule_model.decide(self.agents, self.current_time)
            for agent, task in selected_task:
                if task is not None:
                    coworker = self.agents[self.agents.index(agent) - 1]
                    agent.execute_task(task=task, job=self.job, current_time=self.current_time, coworker=coworker)
                    if online_plot:
                        self.plot.update_info(agent, start=True)

            self.current_time += 1
            # self.shift_schedule()

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
        self.schedule_model.print_schedule()
        logging.info('___________________________________')
        logging.info(f'SIMULATION TOTAL TIME: {time.time() - self.start_time}')
        self.output_data.append(self.schedule_as_dict(hierarchy=True))
        with open(initial_and_final_schedule, 'w') as f:
            json.dump(self.output_data, f, indent=4)

        if experiments:
            statistics = {}
            statistics['makespan'] = [self.job.predicted_makespan, self.job.get_current_makespan()]
            statistics['rejection number'] = len(self.agents[1].rejection_tasks)
            statistics['runtime'] = [self.schedule_model.rescheduling_run_time, self.schedule_model.evaluation_run_time]
            return self.output_data, statistics

    def schedule_as_dict(self, hierarchy=False):
        """
        Returns the current schedule as a dictionary.
        """
        if hierarchy:
            output = {'Human': [], 'Robot': []}
            for task in self.job.task_sequence:
                output[task.agent].append(task.as_dict())
        else:
            output = {
                "Status": [],
                "Start": [],
                "End": [],
                "Agent": [],
                "ID": [],
                "Conditions": [],
                "Object": [],
                "Place": [],
                "Universal": []
            }
            for agent in self.agents:
                for task in agent.tasks_as_dict():
                    output['Status'].append(copy.deepcopy(task['Status']))
                    output['Start'].append(copy.deepcopy(task['Start']))
                    output['End'].append(copy.deepcopy(task['Finish'][0]))
                    output['ID'].append(copy.deepcopy(task['ID']))
                    output['Conditions'].append(copy.deepcopy(task['Conditions']))
                    output['Object'].append(copy.deepcopy(task['Action']['Object']))
                    output['Place'].append(copy.deepcopy(task['Action']['Place']))
                    output['Universal'].append(copy.deepcopy(task['Universal']))
                    output['Agent'].append(copy.deepcopy(task['Agent']))
        return output
