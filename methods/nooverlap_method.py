"""
This class create model and solve methods problem.

@author: Marina Ionova, student of Cybernetics and Robotics at the CTU in Prague
@contact: marina.ionova@cvut.cz
"""
import numpy as np
from typing import Dict
from methods.scheduling_split_tasks import Schedule
from control.agent_and_task_states import AgentState, TaskState
from ortools.sat.python import cp_model
from methods.solver import Solver
import collections
import logging
import copy

LAMBDA = 1


class NoOverlapSchedule(Schedule):
    """
    A class for generating and managing schedules for a given job.

    :param job: Job for which schedule is to be generated.
    :type job: Job
    """
    def __init__(self, job, seed):
        super().__init__(job, seed)
        self.all_agents = [0, 1]
        self.task_intervals: Dict[int, cp_model.IntervalVar] = {}
        self.task_assignment_var: Dict[int, cp_model.IntVar] = {}
        self.task_assignment = self.get_task_assignment_list()
        self.dependencies = self.get_dependencies_list()
        self.agent_mapping = {'Robot': 0, 'Human': 1, 0: 'Robot', 1: 'Human'}

    def set_variables(self):
        """
        Sets constraints for schedule
        """
        # Named tuple to store information about created variables.
        task_info = collections.namedtuple('task_info', 'start end agent interval')


        number_agents = len(self.task_duration)
        no_overlap_task_intervals = []
        PAIRWISE = False

        for i, task in enumerate(self.job.task_sequence):
            # set task intervals and duration
            s = self.model.NewIntVar(0, self.horizon, f"task_{task.id}_start")
            e = self.model.NewIntVar(0, self.horizon, f"task_{task.id}_end")
            d = self.model.NewIntVar(0, self.horizon, f"task_{task.id}_duration")
            self.task_intervals[task.id] = self.model.NewIntervalVar(s, d, e, f"task_interval_{task.id}")

            # create decision variable for task agent assignment
            self.task_assignment_var[task.id] = self.model.NewIntVar(0, number_agents - 1, f"task_{task.id}_assignment")
            # constrain the allowed values according to the given data
            self.model.AddAllowedAssignments(variables=[self.task_assignment_var[task.id]],
                                        tuples_list=[tuple([v]) for v in self.task_assignment[task.id]])

            # set duration of task based on the agent
            for agent in self.all_agents:
                agent_bool = self.model.NewBoolVar(f'agent_{agent}_duration_{self.task_duration[agent][task.id][0]}_for_task_{task.id}')
                self.model.Add(self.task_assignment_var[task.id] == agent).OnlyEnforceIf(agent_bool)
                self.model.Add(self.task_assignment_var[task.id] != agent).OnlyEnforceIf(agent_bool.Not())
                dur = self.task_duration[agent][task.id][0]
                self.model.Add(self.task_intervals[task.id].SizeExpr() == dur).OnlyEnforceIf(agent_bool)

            if not PAIRWISE:
                # make an optional interval per task and agent.
                presence = []

                for agent in self.all_agents:
                    offset = 1000 * agent
                    is_present = self.model.NewBoolVar(f"is_present_task_{task.id}_agent_{agent}")
                    presence.append(is_present)
                    # define the optional interval using the same integer vars as the main task interval + an offset that sets intervals assigned to different agents appart.
                    no_overlap_task_intervals.append(
                        self.model.NewOptionalIntervalVar(s + offset, d, e + offset, is_present=is_present,
                                                     name=f"phantom_task_interval_{task.id}_agent_{agent}"))

                # enforce that exactly one of the optional intervals is present according to the task assignment var
                self.model.Add(sum(presence) == 1)
                self.model.AddElement(index=self.task_assignment_var[task.id], variables=presence, target=True)

        if not PAIRWISE:
            # use the global NoOverlap constraint on all these intervals to prevent tasks assigned to the same agent from overlapping.
            self.model.AddNoOverlap(no_overlap_task_intervals)


        # add task dependency constraints
        for task, dependency_lst in self.dependencies.items():
            for dep in dependency_lst:
                self.model.Add(self.task_intervals[task].StartExpr() > self.task_intervals[dep].EndExpr())

        # formulate objective
        makespan = self.model.NewIntVar(0, self.horizon, "makespan")
        self.model.AddMaxEquality(makespan, [interval.EndExpr() for interval in self.task_intervals.values()])

        self.model.Minimize(makespan)

    def set_constraints(self):
        pass

    def solve(self):

        self.solver = cp_model.CpSolver()
        self.solver.parameters.log_search_progress = True
        self.solver.parameters.max_time_in_seconds = 5
        status = self.solver.Solve(self.model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:

            for task, interval in self.task_intervals.items():
                agent = self.solver.Value(self.task_assignment_var[task])
                assert self.task_duration[agent][task][0] == self.solver.Value(interval.SizeExpr()), f"Duration of task {task} is wrong."
            logging.info("All intervals have the correct duration.")

            for task, agents in self.task_assignment.items():
                assert self.solver.Value(self.task_assignment_var[task]) in agents
            assert len(self.task_assignment_var) == len(self.task_assignment)
            logging.info("All task assignments are valid.")

            for task, deps in self.dependencies.items():
                start = self.solver.Value(self.task_intervals[task].StartExpr())
                ends = [self.solver.Value(self.task_intervals[dep].EndExpr()) for dep in self.dependencies[task]]

                if len(ends) == 0:
                    continue
                assert start >= max(ends), f"dependency graph violation for task {task}."
            logging.info("The solution is valid. No dependency violation")

        elif status == cp_model.INFEASIBLE:
            logging.error("No solution found")
        else:
            logging.error("Something is wrong, check the status and the log of the solve")

        makespan = 0
        output = {'Human' : [], 'Robot': []}
        for task in self.job.task_sequence:
            task.agent = 'Robot' if self.solver.Value(self.task_assignment_var[task.id]) == 0 else 'Human'
            task.start = self.solver.Value(self.task_intervals[task.id].StartExpr())
            task.finish = [self.solver.Value(self.task_intervals[task.id].EndExpr()),
                           self.task_duration[task.agent][task.id][1],
                           self.task_duration[task.agent][task.id][2],
                           self.task_duration[task.agent][task.id][3],
                           ]
            output[task.agent].append(task)
            makespan = task.finish[0] if task.finish[0] > makespan else makespan

        output['Human'].sort(key=lambda x: x.start)
        output['Robot'].sort(key=lambda x: x.start)

        return output, makespan

    def get_task_assignment_list(self):
        task_assignment = {}
        for task in self.job.task_sequence:
            if not task.universal and task.agent == 'Robot':
                task_assignment[task.id] = [0]
            elif not task.universal and task.agent == 'Human':
                task_assignment[task.id] = [1]
            else:
                task_assignment[task.id] = [0, 1]
        return task_assignment

    def get_dependencies_list(self):
        dependencies = {}
        for task in self.job.task_sequence:
            dependencies[task.id] = task.conditions
        return dependencies






