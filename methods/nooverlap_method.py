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
        self.duration_constraints = {}
        self.dependencies = self.get_dependencies_list()
        self.agent_mapping = {'Robot': 0, 'Human': 1, 0: 'Robot', 1: 'Human'}
        self.fix_agent = {}
        self.allowedAgents = {}
        self.dependencies_constraints = {}

    def set_variables(self):
        """
        Sets constraints for schedule
        """

        number_agents = len(self.all_agents)
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
            self.allowedAgents[task.id] = self.model.AddAllowedAssignments(variables=[self.task_assignment_var[task.id]],
                                                                           tuples_list=[tuple([v]) for v in self.task_assignment[task.id]])

            # set duration of task based on the agent
            duration = []
            for agent in self.all_agents:
                duration.append(self.task_duration[agent][task.id][0])
            self.duration_constraints[task.id] = self.model.AddElement(index=self.task_assignment_var[task.id],
                                                                       variables=duration,
                                                                       target=self.task_intervals[task.id].SizeExpr())

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
            self.dependencies_constraints[task] = {}
            for dep in dependency_lst:
                logging.debug(f'Task {task} dependency {dep}')
                self.dependencies_constraints[task][dep] = self.model.Add(self.task_intervals[task].StartExpr() > self.task_intervals[dep].EndExpr())

        # formulate objective
        makespan = self.model.NewIntVar(0, self.horizon, "makespan")
        self.model.AddMaxEquality(makespan, [interval.EndExpr() for interval in self.task_intervals.values()])

        self.model.Minimize(makespan)

    def set_constraints(self):
        pass

    def solve(self):
        self.status = self.solver.Solve(self.model)

        if self.status == cp_model.OPTIMAL or self.status == cp_model.FEASIBLE:
            for task, interval in self.task_intervals.items():
                agent = self.solver.Value(self.task_assignment_var[task])
                # assert self.task_duration[agent][task][0] == self.solver.Value(interval.SizeExpr()), f"Duration of task {task} is wrong."
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
                # assert start >= max(ends), f"dependency graph violation for task {task}."
            logging.info("The solution is valid. No dependency violation")

        elif self.status == cp_model.INFEASIBLE:
            self.job.__str__()
            new_model = self.model.Clone()
            new_model.ClearObjective()

            new_model.AddAssumptions(self.assumptions.values())
            solver = cp_model.CpSolver()
            solver.parameters.num_search_workers = 1
            solver.parameters.search_branching = cp_model.AUTOMATIC_SEARCH
            solver.parameters.log_search_progress = True
            status = solver.Solve(new_model)

            assumptions = copy.copy(self.assumptions)

            for i in solver.SufficientAssumptionsForInfeasibility():
                del assumptions[self.model.VarIndexToVarProto(i).name]

            new_model.ClearAssumptions()
            new_model.AddAssumptions(assumptions.values())
            solver = cp_model.CpSolver()
            solver.parameters.num_search_workers = 1
            solver.parameters.search_branching = cp_model.AUTOMATIC_SEARCH
            solver.parameters.log_search_progress = True
            status = solver.Solve(new_model)

            for i in solver.SufficientAssumptionsForInfeasibility():
                print(self.model.VarIndexToVarProto(i).name)
            print(self.model.Validate())

            logging.error("No solution found")
            exit(2)
        else:
            print(self.model.Validate())
            logging.error(f"Something is wrong, status {self.solver.StatusName(self.status)} and the log of the solve")
            exit(2)

        makespan = 0
        output = {'Human': [], 'Robot': []}
        for task in self.job.task_sequence:
            task.agent = self.agent_mapping[self.solver.Value(self.task_assignment_var[task.id])]

            if (task.state == TaskState.UNAVAILABLE) or (task.state == TaskState.AVAILABLE) or (task.state is None):
                task.start = self.solver.Value(self.task_intervals[task.id].StartExpr())
                task.finish = [self.solver.Value(self.task_intervals[task.id].EndExpr()),
                               self.task_duration[task.agent][task.id][1],
                               self.task_duration[task.agent][task.id][2],
                               self.task_duration[task.agent][task.id][3],
                               ]
            elif task.state == TaskState.InProgress:
                self.job.task_sequence[task.id].finish = \
                    [task.start + self.task_duration[task.agent][task.id][0],
                     self.task_duration[task.agent][task.id][1],
                     self.task_duration[task.agent][task.id][2], self.task_duration[task.agent][task.id][3]]
            else:
                task.start = self.solver.Value(self.task_intervals[task.id].StartExpr())
                task.finish[0] = self.solver.Value(self.task_intervals[task.id].EndExpr())


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

    def fix_agents_var(self):
        """
        Sets allocated agents variable as hard constraints.
        """
        for task in self.job.task_sequence:
            if task.universal:
                idx = self.allowedAgents[task.id].Index()
                self.model.Proto().constraints[idx].Clear()
                # self.model.Proto().constraints[idx].remove(self.allowedAgents[task.id].Proto())
                self.allowedAgents[task.id] = self.model.AddAllowedAssignments(
                    variables=[self.task_assignment_var[task.id]],
                    tuples_list=[tuple([self.agent_mapping[task.agent]])])

    def refresh_variables(self, current_time):
        """
        Changes the variable domains according to what is happening to update the schedule.
        """
        for i, task in enumerate(self.job.task_sequence):
            if (task.id not in self.tasks_with_final_var) and (task.state in [TaskState.InProgress, TaskState.COMPLETED]):

                if task.state == TaskState.COMPLETED:
                    task_duration = int(task.finish[0]) - int(task.start)

                    idx = self.duration_constraints[task.id].Index()
                    self.model.Proto().constraints[idx].Clear()

                    self.model.Proto().variables[self.task_intervals[task.id].EndExpr().Index()].domain[:] = []
                    self.model.Proto().variables[self.task_intervals[task.id].EndExpr().Index()].domain.extend(
                        cp_model.Domain(int(task.finish[0]), int(task.finish[0])).FlattenedIntervals())

                    self.tasks_with_final_var.append(task.id)
                    logging.debug(f'Task {task.id}, new var: finish {task.finish[0]}, duration {task_duration}')
                else:
                    # Change start var
                    if task.finish[0] < current_time:
                        task_duration = current_time - task.start

                        idx = self.duration_constraints[task.id].Index()
                        self.model.Proto().constraints[idx].Clear()
                        # self.duration_constraints[task.id] = self.model.AddElement(index=0,
                        #                                                            variables=[task_duration],
                        #                                                            target=self.task_intervals[
                        #                                                                task.id].SizeExpr())
                        self.model.Proto().variables[self.task_intervals[task.id].EndExpr().Index()].domain[:] = []
                        self.model.Proto().variables[self.task_intervals[task.id].EndExpr().Index()].domain.extend(
                            cp_model.Domain(int(current_time), int(current_time)).FlattenedIntervals())

                self.model.Proto().variables[self.task_intervals[task.id].StartExpr().Index()].domain[:] = []
                self.model.Proto().variables[self.task_intervals[task.id].StartExpr().Index()].domain.extend(
                    cp_model.Domain(int(task.start), int(task.start)).FlattenedIntervals())
                logging.debug(f'Task {task.id}, new var: start {task.start}')

                for dep in self.job.task_sequence:
                    if task.id in dep.conditions:
                        logging.debug(f'Task {dep.id} dependency {task.id}')
                        idx = self.dependencies_constraints[dep.id][task.id].Index()
                        self.model.Proto().constraints[idx].Clear()

            elif task.state == TaskState.AVAILABLE or task.state == TaskState.UNAVAILABLE:
                # Change start var
                self.model.Proto().variables[self.task_intervals[task.id].StartExpr().Index()].domain[:] = []
                self.model.Proto().variables[self.task_intervals[task.id].StartExpr().Index()].domain.extend(
                    cp_model.Domain(int(current_time), self.horizon).FlattenedIntervals())

    def change_agent_in_model(self, task):
        """
        Changes agent variable domain.

        :param task: Task to be redirected to another agent.
        :type task: Task
        """
        idx = self.allowedAgents[task.id].Index()
        self.model.Proto().constraints[idx].Clear()
        # self.model.Proto().constraints.remove(self.allowedAgents[task.id].Proto())
        self.allowedAgents[task.id] = self.model.AddAllowedAssignments(
            variables=[self.task_assignment_var[task.id]],
            tuples_list=[tuple([v]) for v in self.task_assignment[task.id] if v != self.agent_mapping[task.agent]])
        # self.model.Proto().constraints.remove(self.allowedAgents[task.id].Proto())
        # self.model.Proto().constraints.insert(idx, self.allowedAgents[task.id].Proto())

    def set_list_of_possible_changes(self, available_tasks, agent_name, agent_rejection_tasks):
        makespans = []
        for available_task in available_tasks:
            if available_task.id not in agent_rejection_tasks:
                test_model = self.model.Clone()
                task_assignment_var = copy.deepcopy(self.task_assignment_var)
                idx = self.allowedAgents[available_task.id].Index()
                test_model.Proto().constraints[idx].Clear()
                test_model.AddAllowedAssignments(
                    variables=[task_assignment_var[available_task.id]],
                    tuples_list=[tuple([self.agent_mapping[agent_name]])])

                solver = cp_model.CpSolver()
                solver.parameters.num_search_workers = 1
                solver.parameters.random_seed = self.seed
                solver.parameters.max_time_in_seconds = 10.0
                solver.parameters.enumerate_all_solutions = True
                solver.parameters.search_branching = cp_model.AUTOMATIC_SEARCH
                status = solver.Solve(test_model)
                if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                    makespans.append([solver.ObjectiveValue(), available_task])
                    self.evaluation_run_time.append(solver.WallTime())

        if len(makespans) == 0:
            return None
        elif len(makespans) > 1:
            makespans.sort(key=lambda x: x[0])
            return makespans
        else:
            return makespans



