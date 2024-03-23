"""
    ControlLogic class for job execution based on reactive schedule

    @author: Marina Ionova, student of Cybernetics and Robotics at the CTU in Prague
    @contact: marina.ionova@cvut.cz
"""
from visualization import Vis, initial_and_final_schedule_save_file_name, Web_vis
from methods import Solver
from control.agent_and_task_states import AgentState, TaskState
from control.agents import Agent
from control.jobs import Job, Task
import numpy as np
import logging
import json
import time
import copy
import os





class ControlLogic:
    """
    Class that controls the methods and execution of tasks by agents.

    :param case: Case to be executed.
    :type case: str
    """
    def __init__(self, job: Job, agents: list[Agent], method: Solver, **kwargs):
        self.method = method
        self.job = job

        self.agent_list = ['Robot', 'Human']
        self.agents: list[Agent] = agents
        self.current_time = 0
        self.task_finish_time = []
        self.solving_method = method
        self.available_tasks = []
        self.output_data = []
        self.decision_making_duration = []

        self.plot = None
        try:
            self.job.predicted_makespan = self.job.get_current_makespan()
        except TypeError:
            pass
        self.set_task_status()

    def set_task_status(self):
        """
        Sets the status of tasks for each agent.
        """
        for i, task in enumerate(self.job.task_sequence):
            task.state = TaskState.UNAVAILABLE if len(task.conditions) != 0 else TaskState.AVAILABLE

    def get_observation_data(self):
        """
        Checks the progress of each agent's current task and updates the schedule accordingly.
        """
        output = []
        for agent in self.agents:
            if agent.state == AgentState.REJECTION or agent.state == AgentState.ACCEPTANCE:
                pass
            elif agent.state == AgentState.DONE:
                agent.state = AgentState.IDLE
            elif not agent.state == AgentState.IDLE:
                coworker = self.agents[self.agents.index(agent) - 1]
                agent.get_feedback(self.current_time, coworker=coworker)
            try:
                output.append([agent.name, agent.state, agent.current_task, np.array(agent.rejection_tasks)[:, 0]])
            except IndexError:
                output.append([agent.name, agent.state, agent.current_task, []])

        return output

    def run(self, animation:bool=False, online_plot:bool=False, experiments:bool=True, save2file:bool=False, offline_video:bool=False):
        """
        Run the methods simulation.
        """
        try:
            self.output_data.append(self.schedule_as_dict(hierarchy=True))
        except KeyError:
            self.output_data.append([])
        except ValueError:
            self.output_data.append([])

        if online_plot:
            self.plot = Web_vis(data=self.schedule_as_dict(hierarchy=True))

        if offline_video:
            self.plot = Vis(horizon=self.solving_method.horizon)
            self.plot.delete_existing_file()
            self.init_schedule = None
            # self.plot.current_time = self.current_time
            # self.init_schedule = copy.deepcopy(self.schedule_as_dict(hierarchy=True))
            # self.plot.data = [self.init_schedule, self.schedule_as_dict(hierarchy=True)]
            # self.plot.plot_schedule(self.plot.save_path.joinpath(f'{self.current_time:03d}.png'), video=True,
            #                         stat=self.solving_method.get_statistics())
            # # self.plot.data = self.job.__dict__
            # self.plot.save_data()

        start_time = time.perf_counter()

        while True:
            if self.job.progress() == 100:
                break
            logging.debug(f'TIME: {self.current_time}. Progress {self.job.progress()}')

            observation_data = self.get_observation_data()

            # ask planner to decide about next actions for robot and human
            decision_making_start = time.perf_counter()
            try:
                selected_task: dict[str, Task|None] = self.solving_method.decide(observation_data, self.current_time)
            except ValueError as e:
                msg = f"{e} \nsim_seed {self.agents[0].seed}"
                logging.error(msg=msg)
                break
                # raise ValueError(msg)
            self.decision_making_duration.append(time.perf_counter()-decision_making_start)

            for agent in self.agents:
                if agent.state == AgentState.REJECTION:
                    agent.state = AgentState.IDLE
                elif agent.state == AgentState.ACCEPTANCE:
                    agent.state = AgentState.PREPARATION
                if selected_task[agent.name] is None:
                    continue
                
                coworker = self.agents[self.agents.index(agent) - 1]
                agent.execute_task(task=selected_task[agent.name], job=self.job, current_time=self.current_time, coworker=coworker)
                if online_plot:
                    self.plot.update_info(agent, start=True)
            for task in selected_task.values():
                if task is None:
                    continue
                if task.state == TaskState.ASSIGNED:
                    raise ValueError(f"task {task.id} was left in state {task.state}. It will not be possible to execute the task.")


            if online_plot:
                self.plot.current_time = self.current_time
                self.plot.data = self.schedule_as_dict(hierarchy=True)
                self.plot.update_gantt_chart()
                self.plot.update_dependency_graph()
                time.sleep(1)

            if self.current_time == 0:
                try:
                    self.output_data = []
                    self.output_data.append(copy.deepcopy(self.schedule_as_dict(hierarchy=True)))
                except KeyError:
                    self.output_data.append([])
                except ValueError:
                    self.output_data.append([])

            if offline_video:
                # save current state
                if self.current_time == 0:
                    self.init_schedule = copy.deepcopy(self.schedule_as_dict(hierarchy=True))


                if self.plot.current_time + 2 == self.current_time or self.current_time == 0:
                    self.plot.current_time = self.current_time
                    self.plot.data = [self.init_schedule, copy.deepcopy(self.schedule_as_dict(hierarchy=True))]
                    self.plot.plot_schedule(self.plot.save_path.joinpath(f'{self.current_time:03d}.png'), video=True,
                                            stat=self.solving_method.get_statistics())
                    # self.plot.data = self.job.__dict__
                    self.plot.save_data()
            self.current_time += 1

        if offline_video:
            self.plot.current_time = self.current_time
            self.plot.data = [self.init_schedule, copy.deepcopy(self.schedule_as_dict(hierarchy=True))]
            self.plot.plot_schedule(self.plot.save_path.joinpath(f'{self.current_time:03d}.png'), video=True,
                                    stat=self.solving_method.get_statistics())
            # self.plot.data = self.job.__dict__
            self.plot.save_data()
            os.system("ffmpeg -framerate 5 -pattern_type glob -i '*.png' -c:v libx264 -pix_fmt yuv420p simulation.mp4")
            logging.info(f'The video was saved to simulation.mp4')

        logging.info('__________FINAL SCHEDULE___________')
        self.job.__str__()
        logging.info('___________________________________')
        sim_time = time.perf_counter() - start_time
        logging.info(f'SIMULATION TOTAL TIME: {sim_time}')
        self.output_data.append(self.schedule_as_dict(hierarchy=True))
        if animation:
            gantt = Vis(data=self.output_data, from_file=False)
            gantt.plot_schedule(file_name='simulation')
        if save2file:
            with open(initial_and_final_schedule_save_file_name, 'w') as f:
                json.dump(self.output_data, f, indent=4)
            logging.info(f'The data was saved to {initial_and_final_schedule_save_file_name}')

        if experiments:
            statistics = {}
            statistics['sim_time'] = sim_time
            statistics['decision_making_duration'] = self.decision_making_duration
            statistics['initial_makespan'] = self.job.predicted_makespan
            statistics['final_makespan'] = self.job.get_current_makespan()
            statistics['rejection tasks'] = self.agents[1].rejection_tasks
            statistics['solver'] = self.solving_method.get_statistics()
            return self.output_data, statistics

    def schedule_as_dict(self, hierarchy: bool=False) -> dict[str, list[Task]]:
        """
        Returns the current schedule as a dictionary.
        """
        if hierarchy:
            output = {'Human': [], 'Robot': []}
            for task in self.job.task_sequence:
                if len(task.agent) != 1:
                    output[task.agent[0]].append(task.dict())
                output[task.agent[0]].append(task.dict())
        else:
            output = {
                "status": [],
                "start": [],
                "finish": [],
                "agent": [],
                "id": [],
                "conditions": [],
                "object": [],
                "place": [],
                "universal": []
            }
            for agent in self.agents:
                for task in agent.tasks_as_dict():
                    output['Status'].append(copy.deepcopy(task['Status']))
                    output['Start'].append(copy.deepcopy(task['Start']))
                    output['Finish'].append(copy.deepcopy(task['Finish'][0]))
                    output['ID'].append(copy.deepcopy(task['ID']))
                    output['Conditions'].append(copy.deepcopy(task['Conditions']))
                    output['Object'].append(copy.deepcopy(task['Action']['Object']))
                    output['Place'].append(copy.deepcopy(task['Action']['Place']))
                    output['Universal'].append(copy.deepcopy(task['Universal']))
                    output['Agent'].append(copy.deepcopy(task['Agent']))
        return output


def make_control_logic(job: Job, agents: list[Agent], method: Solver) -> ControlLogic:
    pass
