"""
This class create model and solve methods problem.

@author: Marina Ionova, student of Cybernetics and Robotics at the CTU in Prague
@contact: marina.ionova@cvut.cz
"""
import numpy as np
from control.agent_and_task_states import AgentState, TaskState
from ortools.sat.python import cp_model
from control.jobs import Job
from methods.solver import Solver
import collections
import logging
import copy

LAMBDA = 1


START_AVAILABLE_TASKS = True

class Schedule(Solver):
    """
    A class for generating and managing schedules for a given job.

    :param job: Job for which schedule is to be generated.
    :type job: Job
    """
    def __init__(self, job: Job, seed: int):
        self.COUNTER = 0
        self.job = job
        self.model: cp_model.CpModel = cp_model.CpModel()
        self.schedule = {}
        self.current_makespan = None
        self.solver = 0
        self.status = None
        self.all_tasks = {}
        self.current_capacity = {}
        self.horizon = 0
        self.horizon_ceil_1000 = 1000
        self.duration = [0] * self.job.task_number
        self.model_agent = [0] * self.job.task_number
        self.human_task_bool = [True] * self.job.task_number
        self.task_duration = {"Human": [], "Robot": []}
        self.start_var = [0] * self.job.task_number
        self.end_var = [0] * self.job.task_number
        self.tasks_with_final_var = []
        self.tasks_with_deleted_constraints = []
        self.duration_constraints = [[0, 0] for i in range(self.job.task_number)]
        self.fix_agent = [0] * self.job.task_number
        self.border_constraints = [[[0, 0, 0, 0, 0]] * self.job.task_number] * self.job.task_number

        self.rescheduling_run_time = []
        self.evaluation_run_time = []
        self.soft_constr = [0] * self.job.task_number

        self.seed = seed
        self.rand = np.random.default_rng(seed=self.seed)
        self.task_duration = self.job.task_duration(rand_gen=self.rand)

        self.solver = None

        self.assumptions = {}

    def set_variables(self):
        """
        Sets constraints for schedule
        """
        # Named tuple to store information about created variables.
        task_info = collections.namedtuple('task_info', 'start end agent interval')

        for i, task in enumerate(self.job.task_sequence):
            self.human_task_bool[i] = self.model.NewBoolVar(f"task_{task.id}_4_human")
            suffix = f'_{task.id}'

            # condition for different agent
            self.start_var[i] = self.model.NewIntVar(0, self.horizon, 'start' + suffix)
            self.end_var[i] = self.model.NewIntVar(0, self.horizon, 'end' + suffix)

            self.duration[i] = self.model.NewIntVarFromDomain(
                cp_model.Domain.FromIntervals(
                    [[self.task_duration["Human"][task.id][0]], [self.task_duration["Robot"][task.id][0]]]),
                'duration' + suffix)

            if task.agent == "Human":
                self.model.Add(self.human_task_bool[i] == True)
            elif task.agent == "Robot":
                self.model.Add(self.human_task_bool[i] == False)
            else:
                prob = int(LAMBDA*task.rejection_prob*10)
                self.soft_constr[i] = self.model.NewIntVar(0, prob, 'rejection'+suffix)
                self.model.Add(self.soft_constr[i] == prob).OnlyEnforceIf(self.human_task_bool[i])
                self.model.Add(self.soft_constr[i] == 0).OnlyEnforceIf(self.human_task_bool[i].Not())


            interval_var = self.model.NewIntervalVar(self.start_var[i], self.duration[i], self.end_var[i],
                                                     'interval' + suffix)
            self.all_tasks[task.id] = task_info(start=self.start_var[i],
                                                end=self.end_var[i],
                                                agent=self.human_task_bool[i],
                                                interval=interval_var)

    def set_constraints(self):
        """
        Sets constraints for schedule
        """
        # Precedences inside a job.
        for i, task in enumerate(self.job.task_sequence):
            self.duration_constraints[i][0] = self.model.Add(self.duration[i] == self.task_duration["Human"][task.id][0]) \
                .OnlyEnforceIf(self.human_task_bool[i])
            self.duration_constraints[i][1] = self.model.Add(self.duration[i] == self.task_duration["Robot"][task.id][0]) \
                .OnlyEnforceIf(self.human_task_bool[i].Not())

            self.model.Add(self.all_tasks[task.id].end > self.all_tasks[task.id].start)

            # Precedence constraints, which prevent dependent tasks for from overlapping in time.
            # No overlap constraints, which prevent tasks for the same agent from overlapping in time.
            for j in range(self.job.task_number):
                if self.job.task_sequence[j].id != task.id:
                    same_agent = self.model.NewBoolVar(f"same_agent_4_tasks_{task.id}_{j}")
                    self.model.Add(self.human_task_bool[i] == self.human_task_bool[j]).OnlyEnforceIf(same_agent)
                    self.model.Add(self.human_task_bool[i] != self.human_task_bool[j]).OnlyEnforceIf(same_agent.Not())

                    dependent_task_id = self.job.task_sequence[j].id
                    condition = self.model.NewBoolVar(f"{j}_depend_on_{i}")
                    if task.id in self.job.task_sequence[j].conditions:
                        logging.debug(f'Create condition for task {j} <- task {i}')
                        self.model.Add(condition == True)
                    else:
                        self.model.Add(condition == False)

                    # If not conditions and same agents
                    after = self.model.NewBoolVar(f"{j}_after_{task.id}")
                    self.border_constraints[i][j][0] = self.model.Add(self.all_tasks[j].start >= self.all_tasks[i].end). \
                        OnlyEnforceIf([condition.Not(), same_agent, after])
                    self.border_constraints[i][j][1] = self.model.Add(self.all_tasks[j].end <= self.all_tasks[i].start). \
                        OnlyEnforceIf([condition.Not(), same_agent, after.Not()])

                    # If conditions and same agents
                    self.border_constraints[i][j][2] = self.model.Add(after == True). \
                        OnlyEnforceIf([condition, same_agent])
                    # self.border_constraints[i][j][2] = self.model.Add(
                    #     self.all_tasks[dependent_task_id].start >= self.all_tasks[task.id].end). \
                    #     OnlyEnforceIf([condition, same_agent])

                    # If conditions and not same agents
                    k = self.model.NewIntVar(0, 1000, f'overlap_offset_{i}_{j}')
                    k1 = self.task_duration["Human"][task.id][1] + self.task_duration["Human"][task.id][2] - \
                         self.task_duration["Robot"][dependent_task_id][1]
                    k2 = self.task_duration["Robot"][task.id][1] + self.task_duration["Robot"][task.id][2] - \
                         self.task_duration["Human"][dependent_task_id][1]

                    # k1 = self.task_duration["Human"][task.id][3] + self.task_duration["Robot"][dependent_task_id][1]
                    # k2 = self.task_duration["Robot"][task.id][3] + self.task_duration["Human"][dependent_task_id][1]

                    if k1 < 0:
                        k1 = 0
                    if k2 < 0:
                        k2 = 0

                    self.model.Add(k == k1).OnlyEnforceIf([self.human_task_bool[i], condition])
                    self.model.Add(k == k2).OnlyEnforceIf([self.human_task_bool[i].Not(), condition])
                    logging.debug(f'task i {i}, dt j {j}, k = {k2}')
                    self.border_constraints[i][j][3] = self.model.Add(
                    self.all_tasks[dependent_task_id].end >= self.all_tasks[task.id].end + k) \
                        .OnlyEnforceIf([condition, same_agent.Not()])
                    # self.all_tasks[dependent_task_id].start >= self.all_tasks[task.id].end - k) \

                    self.border_constraints[i][j][4] = self.model.Add(
                    self.all_tasks[dependent_task_id].start >= self.all_tasks[task.id].start) \
                    .OnlyEnforceIf([condition, same_agent.Not()])

        # Makespan objective.
        obj_var = self.model.NewIntVar(0, self.horizon, 'makespan')
        self.model.AddMaxEquality(obj_var, [self.all_tasks[i].end for i, task in enumerate(self.all_tasks)])
        obj_var1 = self.model.NewIntVar(0, self.horizon, 'soft_constrains')
        self.model.AddMaxEquality(obj_var1, self.soft_constr)
        self.model.Minimize(obj_var + obj_var1)

    def refresh_variables(self, current_time):
        """
        Changes the variable domains according to what is happening to update the schedule.
        """
        for i, task in enumerate(self.job.task_sequence):
            if (task.id not in self.tasks_with_final_var) and (task.state in [TaskState.InProgress, TaskState.COMPLETED]):
                if task.state == TaskState.COMPLETED:
                    task_duration = int(task.finish[0]) - int(task.start)
                    self.model.Proto().variables[self.end_var[i].Index()].domain[:] = []
                    self.model.Proto().variables[self.end_var[i].Index()].domain.extend(
                        cp_model.Domain(int(task.finish[0]), int(task.finish[0])).FlattenedIntervals())

                    # Change duration var
                    self.model.Proto().variables[self.duration[i].Index()].domain[:] = []
                    self.model.Proto().variables[self.duration[i].Index()].domain.extend(
                        cp_model.Domain(task_duration, task_duration).FlattenedIntervals())

                    self.tasks_with_final_var.append(task.id)
                    # logging.debug(f'Task {task.id}, new var: finish {task.finish[0]}, duration {task_duration}')
                else:
                    # Change start var
                    if task.finish[0] < current_time:
                        task_duration = current_time - task.start
                        # Change duration var
                        self.model.Proto().variables[self.duration[i].Index()].domain[:] = []
                        self.model.Proto().variables[self.duration[i].Index()].domain.extend(
                            cp_model.Domain(task_duration, task_duration).FlattenedIntervals())
                        # logging.debug(f'Task {task.id}, new var: duration {task_duration}')

                if self.duration_constraints[i][0].Proto() in self.model.Proto().constraints:
                    # logging.debug(f'Duration constraints has been deleted, Task i {i}')
                    for j in range(2):
                        self.model.Proto().constraints.remove(self.duration_constraints[i][j].Proto())
                # Cancel constraints
                for j, dependent_task in enumerate(self.job.task_sequence):
                    for k in range(5):
                        self.border_constraints[i][j][k].Proto().Clear()

                self.model.Proto().variables[self.start_var[i].Index()].domain[:] = []
                self.model.Proto().variables[self.start_var[i].Index()].domain.extend(
                    cp_model.Domain(int(task.start), int(task.start)).FlattenedIntervals())
                # logging.debug(f'Task {task.id}, new var: start {task.start}')

            elif task.state == TaskState.AVAILABLE or task.state == TaskState.UNAVAILABLE:
                # Change start var
                self.model.Proto().variables[self.start_var[i].Index()].domain[:] = []
                self.model.Proto().variables[self.start_var[i].Index()].domain.extend(
                    cp_model.Domain(int(current_time), self.horizon).FlattenedIntervals())

    def set_solver(self):
        # Creates the solver and solve.
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = self.seed
        solver.parameters.max_time_in_seconds = 9.0
        solver.parameters.enumerate_all_solutions = True
        solver.parameters.log_search_progress = True if logging.getLogger().level == 10 else False
        solver.parameters.search_branching = cp_model.AUTOMATIC_SEARCH
        return solver

    def solve(self, current_time):
        """
        Finds schedula and parsers it.

        :return: Schedula as sequence of tasks for each agent
        :rtype agent: dictionary
        """
        self.assigned_jobs = collections.defaultdict(list)
        self.solver = self.set_solver()
        self.status = self.solver.Solve(self.model)

        # Named tuple to manipulate solution information.
        assigned_task_info = collections.namedtuple('assigned_task_info',
                                                    'start end task_id agent')

        if self.status == cp_model.OPTIMAL or self.status == cp_model.FEASIBLE:
            output = {}

            # Create one list of assigned tasks per machine.
            for i, task in enumerate(self.job.task_sequence):
                if task.agent == "Both":
                    if self.solver.Value(self.all_tasks[i].agent):
                        agent = "Human"
                    else:
                        agent = "Robot"
                else:
                    agent = task.agent
                self.assigned_jobs[agent].append(
                    assigned_task_info(start=self.solver.Value(
                        self.all_tasks[i].start),
                        end=[self.solver.Value(
                            self.all_tasks[i].end), self.task_duration[agent][task.id][1],
                        self.task_duration[agent][task.id][2], self.task_duration[agent][task.id][3]],
                        task_id=i,
                        agent=agent))

            makespan = 0
            for agent in self.job.agents:
                # Sort by starting time.
                self.assigned_jobs[agent].sort()
                output[agent] = []
                for assigned_task in self.assigned_jobs[agent]:
                    start = assigned_task.start
                    end = assigned_task.end
                    if end[0] > makespan:
                        makespan = end[0]

                    task = self.job.task_sequence[assigned_task.task_id]
                    self.job.task_sequence[assigned_task.task_id].agent = agent
                    if (task.state == TaskState.UNAVAILABLE) or (task.state == TaskState.AVAILABLE) or (task.state is None):
                        self.job.task_sequence[assigned_task.task_id].start = start
                        self.job.task_sequence[assigned_task.task_id].finish = end
                    elif task.state == TaskState.InProgress:
                        self.job.task_sequence[assigned_task.task_id].finish = \
                            [task.start + self.task_duration[task.agent][task.id][0],
                             self.task_duration[task.agent][task.id][1],
                             self.task_duration[task.agent][task.id][2], self.task_duration[task.agent][task.id][3]]

                    output[agent].append(self.job.task_sequence[assigned_task.task_id])
            self.rescheduling_run_time.append([current_time, self.solver.StatusName(self.status),
                                               self.solver.ObjectiveValue(), self.solver.WallTime()])

            return output, makespan
        else:
            self.model.ExportToFile('model.txt')
            self.job.__str__()
            logging.error(f"Scheduling failed, max self.horizon: {self.horizon} \n")
            exit(2)

    def make_enforcement_assumption(self, name):
        a1 = self.model.NewBoolVar(name)
        self.assumptions[name] = a1

        return a1

    def fix_agents_var(self):
        """
        Sets allocated agents variable as hard constraints.
        """
        for i, task in enumerate(self.job.task_sequence):
            if task.universal:
                if task.agent == "Human":
                    self.fix_agent[i] = self.model.Add(self.human_task_bool[i] == True)
                else:
                    self.fix_agent[i] = self.model.Add(self.human_task_bool[i] == False)

    def change_agent_in_model(self, task):
        """
        Changes agent variable domain.

        :param task: Task to be redirected to another agent.
        :type task: Task
        """
        idx = self.job.task_sequence.index(task)
        self.model.Proto().constraints.remove(self.fix_agent[idx].Proto())
        if task.agent == "Human":
            self.model.Add(self.human_task_bool[idx] == True)
        else:
            self.model.Add(self.human_task_bool[idx] == False)

    def set_max_horizon(self, **kwargs):
        """
        Computes horizon dynamically as the sum of all durations.
        :return:
        """
        if self.horizon == 0 or self.job.task_sequence[0].finish is None:
            for task in self.job.task_sequence:
                if task.universal:
                    self.horizon += max(self.task_duration['Human'][task.id][0], self.task_duration['Robot'][task.id][0])
                else:
                    self.horizon += self.task_duration[task.agent[0]][task.id][0]
        else:
            self.horizon = max(task.finish[0] for task in self.job.task_sequence) + 100

        self.horizon = int(self.horizon)

        self.horizon_ceil_1000 = int(np.ceil(self.horizon*0.001)*1000)

    def prepare(self, **kwargs):
        """
        Creates variables, their domains and constraints in model, then solves it.
        """
        self.set_max_horizon(**kwargs)
        self.set_variables()
        self.set_constraints()
        # self.schedule, self.current_makespan = self.solve(current_time=0)
        # self.print_schedule()
        # self.fix_agents_var()
        # self.print_info()
        return self.schedule

    def set_list_of_possible_changes(self, available_tasks, agent_name, current_time, **kwargs):
        makespans = []
        for available_task in available_tasks:
            test_model = self.model.Clone()
            human_task_bool_copy = copy.deepcopy(self.human_task_bool)
            idx = self.job.get_task_idx(available_task)
            test_model.Proto().constraints.remove(self.fix_agent[idx].Proto())
            if agent_name == "Human":
                test_model.Add(human_task_bool_copy[idx] == True)
            else:
                test_model.Add(human_task_bool_copy[idx] == False)
            solver = cp_model.CpSolver()
            solver.parameters.num_search_workers = 1
            solver.parameters.random_seed = self.seed
            solver.parameters.max_time_in_seconds = 10.0
            solver.parameters.enumerate_all_solutions = True
            solver.parameters.search_branching = cp_model.AUTOMATIC_SEARCH
            status = solver.Solve(test_model)
            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                makespans.append([solver.ObjectiveValue(), available_task])
                self.evaluation_run_time.append([current_time, solver.WallTime()])

        if len(makespans) == 0:
            return None
        elif len(makespans) > 1:
            makespans.sort(key=lambda x: x[0])
            return makespans
        else:
            return makespans

    def decide(self, observation_data, current_time):
        decision = {}
        self.update_tasks_status()
        self.update_schedule(current_time)
        for index, [agent_name, agent_state, agent_current_task, agent_rejection_tasks] in enumerate(observation_data):
            logging.debug(f'TIME: {current_time}. Is {agent_name} available? {agent_state}')
            if agent_state == AgentState.IDLE:
                decision[agent_name] = self.find_task(agent_name, agent_rejection_tasks, current_time)
            elif agent_state == AgentState.REJECTION:
                coworker = observation_data[index - 1]
                self.change_agent(task=agent_current_task, coworker_name=coworker[0], current_time=current_time)
                decision[agent_name] = self.find_task(agent_name, agent_rejection_tasks, current_time)
            elif agent_state == AgentState.DONE:
                decision[agent_name] = self.find_task(agent_name, agent_rejection_tasks, current_time)
            else:
                decision[agent_name] = None
        return decision

    def find_task(self, agent_name, agent_rejection_tasks, current_time):
        # find allocated task
        for task in self.schedule[agent_name]:
            if task.state == TaskState.AVAILABLE:
                if START_AVAILABLE_TASKS:
                    return task
                else:
                    if task.start <= current_time:
                        return task

        # If the agent has run out of available tasks in his list, he looks for
        # universal tasks in the list of a colleague that he can perform instead
        # of him to speed up the process.
        possible_tasks = []
        for worker in self.schedule.keys():
            if worker != agent_name:
                for task in self.schedule[worker]:
                    if task.state == TaskState.AVAILABLE and task.universal and task.id not in agent_rejection_tasks:
                        possible_tasks.append(task)

        if len(possible_tasks) != 0:
            # rescheduling estimation
            self.refresh_variables(current_time)
            makespan_and_task = self.set_list_of_possible_changes(possible_tasks, agent_name, current_time)
            if makespan_and_task and makespan_and_task[0][0] < self.current_makespan:
                self.change_agent(task=makespan_and_task[0][1], coworker_name=agent_name, current_time=current_time)
                return makespan_and_task[0][1]

        return None

    # def update_tasks_status(self):
    #     """
    #     Updates the status of tasks based on their dependencies.
    #     """
    #     tasks_list = self.job.get_completed_and_in_progress_task_list()
    #     for task in self.job.task_sequence:
    #         if len(task.conditions) != 0 and task.state == TaskState.UNAVAILABLE:
    #             if set(task.conditions).issubset(tasks_list):
    #                 task.state = TaskState.AVAILABLE

    def update_schedule(self, current_time):
        for agent in self.schedule:
            shift = False
            for task in self.schedule[agent]:
                if task.state == TaskState.InProgress and task.finish[0] < current_time:
                    task.finish[0] = current_time
                    shift = True
                elif shift and (task.state == TaskState.UNAVAILABLE or task.state == TaskState.AVAILABLE):
                    task.start += 1
                    task.finish[0] += 1


    def change_agent(self, task, coworker_name, current_time):
        """
        Changes the agent assigned to a task.

        :param task: Task to change agent for.
        :type task: Task
        :type current_agent: Agent
        """
        assert isinstance(coworker_name, str)

        self.change_agent_in_model(task)
        self.job.change_agent(task.id, new_agent_name=coworker_name)

        self.refresh_variables(current_time)
        self.schedule, self.current_makespan = self.solve(current_time)
        self.print_info()

        logging.info('____RESCHEDULING______')
        self.print_schedule()
        logging.info('______________________')

    def get_statistics(self):
        return [self.rescheduling_run_time, self.evaluation_run_time]

    def print_schedule(self):
        logging.info("____________________________")
        logging.info("INFO: Task distribution")
        logging.info("Robot")
        for task in self.schedule["Robot"]:
            task.__str__()
        logging.info("Human")
        for task in self.schedule["Human"]:
            task.__str__()
        logging.info("____________________________")

    def print_info(self):
        """
        Prints basic info about solution and solving process.
        """
        logging.info('Solve status: %s' % self.solver.StatusName(self.status))
        logging.info('Optimal objective value: %i' % self.solver.ObjectiveValue())
        logging.info('Statistics')
        logging.info('  - conflicts : %i' % self.solver.NumConflicts())
        logging.info('  - branches  : %i' % self.solver.NumBranches())
        logging.info('  - wall time : %f s' % self.solver.WallTime())


def schedule_as_dict(schedule):
    schedule_as_dict = {'Human': [], 'Robot': []}
    for agent in ["Human", "Robot"]:
        for task in schedule[agent]:
            schedule_as_dict[agent].append(task.as_dict())
    return schedule_as_dict
