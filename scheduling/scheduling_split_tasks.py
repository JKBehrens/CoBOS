"""
This class create model and solve scheduling problem.

@author: Marina Ionova, student of Cybernetics and Robotics at the CTU in Prague
@contact: marina.ionova@cvut.cz
"""
from simulation.sim import set_task_time
from ortools.sat.python import cp_model
import collections
import logging
import copy

LAMBDA = 1


class Schedule:
    """
    A class for generating and managing schedules for a given job.

    :param job: Job for which schedule is to be generated.
    :type job: Job
    """
    def __init__(self, job):
        self.COUNTER = 0
        self.job = job
        self.model = cp_model.CpModel()
        self.solver = 0
        self.status = None
        self.all_tasks = {}
        self.current_capacity = {}
        self.horizon = 0
        self.duration = [0] * self.job.task_number
        self.model_agent = [0] * self.job.task_number
        self.human_task_bool = [True] * self.job.task_number
        self.task_duration = {"Human": [], "Robot": []}
        self.start_var = [0] * self.job.task_number
        self.end_var = [0] * self.job.task_number
        self.tasks_with_final_var = []
        self.duration_constraints = [[0, 0] for i in range(self.job.task_number)]
        self.fix_agent = [0] * self.job.task_number
        self.border_constraints = [[[0, 0, 0, 0]] * self.job.task_number] * self.job.task_number

        self.rescheduling_run_time = []
        self.evaluation_run_time = []
        self.soft_constr = [0] * self.job.task_number

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
                    [[self.task_duration["Human"][i][0]], [self.task_duration["Robot"][i][0]]]),
                'duration' + suffix)

            if task.agent == "Human":
                self.model.Add(self.human_task_bool[i] == True)
            elif task.agent == "Robot":
                self.model.Add(self.human_task_bool[i] == False)
            else:
                prob = int(LAMBDA*task.get_reject_prob()*10)
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
            self.duration_constraints[i][0] = self.model.Add(self.duration[i] == self.task_duration["Human"][i][0]) \
                .OnlyEnforceIf(self.human_task_bool[i])
            self.duration_constraints[i][1] = self.model.Add(self.duration[i] == self.task_duration["Robot"][i][0]) \
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
                    # print(task.id, self.job.task_sequence[j].id, self.job.task_sequence[j].conditions)
                    if task.id in self.job.task_sequence[j].conditions:
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
                    self.border_constraints[i][j][2] = self.model.Add(
                        self.all_tasks[dependent_task_id].start >= self.all_tasks[task.id].end). \
                        OnlyEnforceIf([condition, same_agent])

                    # If conditions and not same agents
                    k = self.model.NewIntVar(0, 1000, f'overlap_offset_{i}_{j}')
                    k1 = self.task_duration["Human"][i][1] + self.task_duration["Human"][i][2] - \
                         self.task_duration["Robot"][dependent_task_id][1]
                    k2 = self.task_duration["Robot"][i][1] + self.task_duration["Robot"][i][2] - \
                         self.task_duration["Human"][dependent_task_id][1]

                    if k1 < 0:
                        k1 = 0
                    if k2 < 0:
                        k2 = 0

                    logging.debug(f'k1 = {k1}, k2 = {k2}')
                    self.model.Add(k == k1).OnlyEnforceIf([self.human_task_bool[i], condition])
                    self.model.Add(k == k2).OnlyEnforceIf([self.human_task_bool[i].Not(), condition])

                    self.border_constraints[i][j][3] = self.model.Add(
                        self.all_tasks[dependent_task_id].end >= self.all_tasks[task.id].end + k) \
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
            if (task.id not in self.tasks_with_final_var) and (task.status in [1, 2]):
                if task.status == 2:
                    task_duration = int(task.finish[0]) - int(task.start)
                    self.model.Proto().variables[self.end_var[i].Index()].domain[:] = []
                    self.model.Proto().variables[self.end_var[i].Index()].domain.extend(
                        cp_model.Domain(int(task.finish[0]), int(task.finish[0])).FlattenedIntervals())

                    # Change duration var
                    self.model.Proto().variables[self.duration[i].Index()].domain[:] = []
                    self.model.Proto().variables[self.duration[i].Index()].domain.extend(
                        cp_model.Domain(task_duration, task_duration).FlattenedIntervals())

                    self.tasks_with_final_var.append(task.id)
                else:
                    # Change start var
                    if task.finish < current_time:
                        task_duration = current_time - task.start
                        # Change duration var
                        self.model.Proto().variables[self.duration[i].Index()].domain[:] = []
                        self.model.Proto().variables[self.duration[i].Index()].domain.extend(
                            cp_model.Domain(task_duration, task_duration).FlattenedIntervals())
                if self.duration_constraints[i][0].Proto() in self.model.Proto().constraints:
                    for j in range(2):
                        self.model.Proto().constraints.remove(self.duration_constraints[i][j].Proto())
                # Cancel constraints
                for j in range(self.job.task_number):
                    for k in range(4):
                        if not isinstance(self.border_constraints[i][j][k], int) and \
                                self.border_constraints[i][j][k].Proto() in self.model.Proto().constraints:
                            logging.debug(f'Constraints has been deleted, Task{task.id}')
                            self.model.Proto().constraints.remove(self.border_constraints[i][j][k].Proto())

                self.model.Proto().variables[self.start_var[i].Index()].domain[:] = []
                self.model.Proto().variables[self.start_var[i].Index()].domain.extend(
                    cp_model.Domain(int(task.start), int(task.start)).FlattenedIntervals())

            elif task.status == 0 or task.status == -1:
                # Change start var
                self.model.Proto().variables[self.start_var[i].Index()].domain[:] = []
                self.model.Proto().variables[self.start_var[i].Index()].domain.extend(
                    cp_model.Domain(int(current_time), self.horizon).FlattenedIntervals())

    def solve(self):
        """
        Finds schedula and parsers it.

        :return: Schedula as sequence of tasks for each agent
        :rtype agent: dictionary
        """
        self.assigned_jobs = collections.defaultdict(list)
        # Creates the solver and solve.
        self.solver = cp_model.CpSolver()
        self.solver.parameters.random_seed = 73
        self.solver.parameters.max_time_in_seconds = 10.0
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
                        end=self.solver.Value(
                            self.all_tasks[i].end),
                        task_id=i,
                        agent=agent))

            for agent in self.job.agents:
                # Sort by starting time.
                self.assigned_jobs[agent].sort()
                output[agent] = []
                for assigned_task in self.assigned_jobs[agent]:
                    start = assigned_task.start
                    end = assigned_task.end

                    task = self.job.task_sequence[assigned_task.task_id]
                    self.job.task_sequence[assigned_task.task_id].agent = agent
                    if (task.status == -1) or (task.status == 0) or (task.status is None):
                        self.job.task_sequence[assigned_task.task_id].start = start
                        self.job.task_sequence[assigned_task.task_id].finish = end
                    elif task.status == 1:
                        self.job.task_sequence[assigned_task.task_id].finish = \
                            task.start + self.task_duration[task.agent][task.id][0]

                    output[agent].append(self.job.task_sequence[assigned_task.task_id])
            self.rescheduling_run_time.append([self.solver.StatusName(self.status),
                                               self.solver.ObjectiveValue(), self.solver.WallTime()])
            return output
        else:
            logging.error(f"Scheduling failed, max self.horizon: {self.horizon} \n")
            self.job.__str__()
            exit()

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

    def set_new_agent(self, task):
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
        self.horizon = 0
        self.set_duration_of_all_tasks(**kwargs)
        for i, task in enumerate(self.job.task_sequence):
            if task.universal:
                self.horizon += max(self.task_duration['Human'][i][0], self.task_duration['Robot'][i][0])
            else:
                self.horizon += self.task_duration[task.agent][i][0]
        self.horizon = int(self.horizon)

    def set_duration_of_all_tasks(self, **kwargs):
        """
        Set durations of all tasks by each agent
        """
        for task in self.job.task_sequence:
            self.task_duration["Human"].append(set_task_time(task, 'Human',  **kwargs))
            self.task_duration["Robot"].append(set_task_time(task, 'Robot',  **kwargs))

    def set_schedule(self, **kwargs):
        """
        Creates variables, their domains and constraints in model, then solves it.
        """
        self.set_max_horizon(**kwargs)
        self.set_variables()
        self.set_constraints()
        schedule = self.solve()
        print_schedule(schedule)
        self.fix_agents_var()
        self.print_info()
        return schedule

    def set_list_of_possible_changes(self, available_tasks, agent):
        makespans = []
        for available_task in available_tasks:
            if available_task.id not in agent.rejection_tasks:
                test_model = copy.deepcopy(self.model)
                human_task_bool_copy = copy.deepcopy(self.human_task_bool)
                idx = self.job.get_task_idx(available_task)
                test_model.Proto().constraints.remove(self.fix_agent[idx].Proto())
                if agent.name == "Human":
                    test_model.Add(human_task_bool_copy[idx] == True)
                else:
                    test_model.Add(human_task_bool_copy[idx] == False)
                solver = cp_model.CpSolver()
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


def print_schedule(schedule):
    logging.info("____________________________")
    logging.info("INFO: Task distribution")
    logging.info("Robot")
    for task in schedule["Robot"]:
        task.__str__()
    logging.info("Human")
    for task in schedule["Human"]:
        task.__str__()
    logging.info("____________________________")
