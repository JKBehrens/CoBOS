"""
This class create model and solve methods problem.

@author: Marina Ionova, student of Cybernetics and Robotics at the CTU in Prague
@contact: marina.ionova@cvut.cz
"""
from typing import Dict, List
from control.jobs import Job
from methods.scheduling_split_tasks import Schedule
from control.agent_and_task_states import AgentState
from control.agent_and_task_states import TaskState
from ortools.sat.python import cp_model
import logging
import statistics

LAMBDA = 1
START_AVAILABLE_TASKS = True

class OverlapSchedule(Schedule):
    """
    A class for generating and managing schedules for a given job.

    :param job: Job for which schedule is to be generated.
    :type job: Job
    """
    def __init__(self, job: Job, seed: int):
        super().__init__(job, seed)
        self.all_agents = [0, 1]
        self.task_intervals: Dict[int, List[cp_model.IntervalVar]] = {}
        self.task_assignment_var: Dict[int, cp_model.IntVar] = {}
        self.task_assignment = self.get_task_assignment_list()
        self.duration_constraints = {}
        self.dependencies = self.get_dependencies_list()
        self.agent_mapping = {'Robot': 0, 'Human': 1, 0: 'Robot', 1: 'Human'}
        self.fix_agent = {}
        self.allowedAgents = {}
        self.dependencies_constraints = {}
        self.no_overlap_execution_intervals = []
        self.no_overlap_execution_constraint = None

    @classmethod
    def name(cls) -> str:
        return str(cls).capitalize().split(".")[-1].split("'>")[0]

    def set_variables(self):
        """
        Sets constraints for schedule
        """

        number_agents = len(self.all_agents)
        no_overlap_task_intervals = []

        for i, task in enumerate(self.job.task_sequence):

            # create decision variable for task agent assignment
            self.task_assignment_var[task.id] = self.model.NewIntVar(0, number_agents - 1, f"task_{task.id}_assignment")
            # constrain the allowed values according to the given data
            self.allowedAgents[task.id] = self.model.AddAllowedAssignments(
                variables=[self.task_assignment_var[task.id]],
                tuples_list=[tuple([v]) for v in self.task_assignment[task.id]])

            self.task_intervals[task.id] = []
            self.duration_constraints[task.id] = []
            # Iterate through task phase
            for phase in range(3):
                # set phase interval
                s = self.model.NewIntVar(0, self.horizon, f"task_{task.id}_phase_{phase}_start")
                e = self.model.NewIntVar(0, self.horizon, f"task_{task.id}_phase_{phase}_end")
                d = self.model.NewIntVar(0, self.horizon, f"task_{task.id}_phase_{phase}_duration")
                self.task_intervals[task.id].append(self.model.NewIntervalVar(s, d, e, f"phase_{phase}_task_{task.id}_interval"))

                # set phase duration based on the agent
                phase_durations = []
                for agent in self.all_agents:
                    phase_durations.append(self.task_duration[agent][task.id][phase+1])
                self.duration_constraints[task.id].append(
                    self.model.AddElement(index=self.task_assignment_var[task.id],
                                          variables=phase_durations,
                                          target=self.task_intervals[task.id][phase].SizeExpr()))

            # add no overlap interval for execution phase
            self.no_overlap_execution_intervals.append(self.task_intervals[task.id][1])

            # set phases continuity
            self.model.Add(self.task_intervals[task.id][0].EndExpr() == self.task_intervals[task.id][1].StartExpr())
            self.model.Add(self.task_intervals[task.id][1].EndExpr() == self.task_intervals[task.id][2].StartExpr())

            # set duration of the task
            duration = self.model.NewIntVar(0, self.horizon, f"task_{task.id}_duration")
            self.model.Add(duration == self.task_intervals[task.id][0].SizeExpr() +
                            self.task_intervals[task.id][1].SizeExpr() +
                            self.task_intervals[task.id][2].SizeExpr())
            # make an optional interval per task and agent.
            presence = []

            for agent in self.all_agents:
                offset = self.horizon_ceil_1000 * agent
                is_present = self.model.NewBoolVar(f"is_present_task_{task.id}_agent_{agent}")
                presence.append(is_present)
                # define the optional interval using the same integer vars as the main task interval
                # + an offset that sets intervals assigned to different agents appart.
                no_overlap_task_intervals.append(
                    self.model.NewOptionalIntervalVar(self.task_intervals[task.id][0].StartExpr() + offset,
                                                        duration, self.task_intervals[task.id][2].EndExpr() + offset,
                                                        is_present=is_present,
                                                        name=f"phantom_task_interval_{task.id}_agent_{agent}"))

            # enforce that exactly one of the optional intervals is present according to the task assignment var
            self.model.Add(sum(presence) == 1)
            self.model.AddElement(index=self.task_assignment_var[task.id], variables=presence, target=True)


        # use the global NoOverlap constraint on all these intervals
        # to prevent tasks assigned to the same agent from overlapping.
        self.model.AddNoOverlap(no_overlap_task_intervals)
        self.model.AddNoOverlap(self.no_overlap_execution_intervals)

        # add task dependency constraints
        for task, dependency_lst in self.dependencies.items():
            self.dependencies_constraints[task] = {}
            for dep in dependency_lst:
                logging.debug(f'Task {task} dependency {dep}')
                self.dependencies_constraints[task][dep] = []
                self.dependencies_constraints[task][dep].append(self.model.Add(
                    self.task_intervals[task][1].StartExpr() >= self.task_intervals[dep][1].EndExpr()))
                self.dependencies_constraints[task][dep].append(self.model.Add(
                    self.task_intervals[task][0].StartExpr() >= self.task_intervals[dep][0].StartExpr()))


        # formulate objective
        self.makespan = self.model.NewIntVar(0, self.horizon, "makespan")
        makespan = self.makespan
        intervals = []
        for task in self.job.task_sequence:
            intervals.append(self.task_intervals[task.id][2].EndExpr())
        self.model.AddMaxEquality(makespan, intervals)

        self.model.Minimize(makespan)

    def set_constraints(self):
        pass

    def constraint_debugger(self):
        model: cp_model.CpModel = self.model.Clone()

        for idx, constraint in enumerate(model.Proto().constraints):
            print(f'idx: {idx}, {constraint}')
            if 'interval' in constraint.__str__():
                continue
            model.Proto().constraints[idx].Clear()
            solver = self.set_solver()
            status = solver.Solve(model)
            print(solver.StatusName(status))
            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                break

    def solve(self, current_time):
        self.solver = self.set_solver()
        self.status = self.solver.Solve(self.model)

        if self.status == cp_model.OPTIMAL or self.status == cp_model.FEASIBLE:
            for task in self.job.task_sequence:
                if task.state in [TaskState.COMPLETED, TaskState.InProgress]:
                    if self.job.task_sequence[task.id].finish[0] < current_time and task.state is TaskState.InProgress:
                        assert (current_time - self.job.task_sequence[task.id].start) <= \
                               self.solver.Value(self.task_intervals[task.id][2].EndExpr()) - \
                               self.solver.Value(self.task_intervals[task.id][0].StartExpr()), \
                            f"Duration of task {task.id} is wrong."
                    # else:
                    #     if not (self.job.task_sequence[task.id].finish[0] - self.job.task_sequence[task.id].start) <=\
                    #            self.solver.Value(self.task_intervals[task.id][2].EndExpr()) - \
                    #            self.solver.Value(self.task_intervals[task.id][0].StartExpr()):
                    #         logging.warning(f"Duration of task {task.id} is wrong.")
                else:
                    agent = self.solver.Value(self.task_assignment_var[task.id])
                    assert self.task_duration[agent][task.id][0] == \
                           self.solver.Value(self.task_intervals[task.id][2].EndExpr()) - \
                           self.solver.Value(self.task_intervals[task.id][0].StartExpr()), \
                        f"Duration of task {task.id} is wrong."
            logging.info("All intervals have the correct duration.")

            for task, agents in self.task_assignment.items():
                assert self.solver.Value(self.task_assignment_var[task]) in agents
            assert len(self.task_assignment_var) == len(self.task_assignment)
            logging.info("All task assignments are valid.")

            for task in self.job.task_sequence:
                start = self.solver.Value(self.task_intervals[task.id][1].StartExpr())
                ends = [self.solver.Value(self.task_intervals[dep][1].EndExpr()) for dep in task.conditions]

                if len(ends) == 0:
                    continue
                # assert start >= max(ends), f"dependency graph violation for task {task.id}."
            logging.info("The solution is valid. No dependency violation")

        elif self.status == cp_model.INFEASIBLE:
            logging.error(self.model.Validate())
            logging.error(f"No solution found. case {self.job.case} \nsolver_seed {self.seed}, dist_seed {self.job.seed}")
            # self.constraint_debugger()
            raise ValueError(f"case {self.job.case}, solver_seed {self.seed}, dist_seed {self.job.seed}")
        else:
            logging.error(self.model.Validate())
            logging.error(f"Something is wrong, status {self.solver.StatusName(self.status)} and the log of the solve")
            raise ValueError(f"case {self.job.case}, solver_seed {self.seed}, dist_seed {self.job.seed}")

        makespan = 0
        output = {'Human': [], 'Robot': []}
        for task in self.job.task_sequence:
            task.agent = [self.agent_mapping[self.solver.Value(self.task_assignment_var[task.id])]]

            if (task.state == TaskState.UNAVAILABLE) or (task.state == TaskState.AVAILABLE) or (task.state is None):
                task.start = self.solver.Value(self.task_intervals[task.id][0].StartExpr())
                task.finish = [self.solver.Value(self.task_intervals[task.id][2].EndExpr()),
                               self.solver.Value(self.task_intervals[task.id][0].SizeExpr()),
                               self.solver.Value(self.task_intervals[task.id][1].SizeExpr()),
                               self.solver.Value(self.task_intervals[task.id][2].SizeExpr()),
                               ]
            elif task.state == TaskState.InProgress:
                self.job.task_sequence[task.id].finish = \
                    [task.start + self.task_duration[task.agent[0]][task.id][0],
                     self.task_duration[task.agent[0]][task.id][1],
                     self.task_duration[task.agent[0]][task.id][2], self.task_duration[task.agent[0]][task.id][3]]
            else:
                task.start = self.solver.Value(self.task_intervals[task.id][0].StartExpr())
                task.finish[0] = self.solver.Value(self.task_intervals[task.id][2].EndExpr())

            output[task.agent[0]].append(task)
            makespan = task.finish[0] if task.finish[0] > makespan else makespan

        solver_info = self.get_solver_info_as_dict(self.solver)
        solver_info['current_time'] = current_time
        self.rescheduling_run_time.append(solver_info)

        output['Human'].sort(key=lambda x: x.start)
        output['Robot'].sort(key=lambda x: x.start)
        return output, makespan

    @staticmethod
    def get_solver_info_as_dict(solver: cp_model.CpSolver):
        return {
            'status': solver.StatusName(solver.ResponseProto().status),
            'objective_value': solver.ResponseProto().objective_value,
            'best_objective_bound': solver.ResponseProto().best_objective_bound,
            'inner_objective_lower_bound': solver.ResponseProto().inner_objective_lower_bound,
            'num_integers': solver.ResponseProto().num_integers,
            'num_booleans': solver.ResponseProto().num_booleans,
            'num_conflicts': solver.ResponseProto().num_conflicts,
            'num_binary_propagations': solver.ResponseProto().num_binary_propagations,
            'num_integer_propagations': solver.ResponseProto().num_integer_propagations,
            'num_lp_iterations': solver.ResponseProto().num_lp_iterations,
            'wall_time': solver.ResponseProto().wall_time,
            'user_time': solver.ResponseProto().user_time,
            'deterministic_time': solver.ResponseProto().deterministic_time,
            'gap_integral': solver.ResponseProto().gap_integral
        }


    def get_task_assignment_list(self):
        task_assignment = {}
        for task in self.job.task_sequence:
            if not task.universal and task.agent[0] == 'Robot':
                task_assignment[task.id] = [0]
            elif not task.universal and task.agent[0] == 'Human':
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
                self.allowedAgents[task.id] = self.model.AddAllowedAssignments(
                    variables=[self.task_assignment_var[task.id]],
                    tuples_list=[tuple([self.agent_mapping[task.agent[0]]])])

    def refresh_variables(self, current_time):
        """
        Changes the variable domains according to what is happening to update the schedule.
        """
        for i, task in enumerate(self.job.task_sequence):
            if task.state in [TaskState.InProgress, TaskState.COMPLETED]:
                idx = self.allowedAgents[task.id].Index()
                self.model.Proto().constraints[idx].Clear()
                self.allowedAgents[task.id] = self.model.AddAllowedAssignments(
                    variables=[self.task_assignment_var[task.id]],
                    tuples_list=[tuple([self.agent_mapping[task.agent[0]]])])

                if task.state == TaskState.COMPLETED:
                    if task.id in self.tasks_with_final_var:
                        continue
                    # delete duration constraints
                    for phase in range(3):
                        idx = self.duration_constraints[task.id][phase].Index()
                        self.model.Proto().constraints[idx].Clear()

                    # set end of interval to current end time of task
                    self.model.Proto().variables[self.task_intervals[task.id][0].EndExpr().Index()].domain[:] = []
                    self.model.Proto().variables[self.task_intervals[task.id][0].EndExpr().Index()].domain.extend(
                        cp_model.Domain(int(task.finish[0]-task.finish[2]-task.finish[3]),
                                        int(task.finish[0]-task.finish[2]-task.finish[3])).FlattenedIntervals())

                    self.model.Proto().variables[self.task_intervals[task.id][1].EndExpr().Index()].domain[:] = []
                    self.model.Proto().variables[self.task_intervals[task.id][1].EndExpr().Index()].domain.extend(
                        cp_model.Domain(int(task.finish[0]-task.finish[3]),
                                        int(task.finish[0]-task.finish[3])).FlattenedIntervals())

                    self.model.Proto().variables[self.task_intervals[task.id][2].EndExpr().Index()].domain[:] = []
                    self.model.Proto().variables[self.task_intervals[task.id][2].EndExpr().Index()].domain.extend(
                        cp_model.Domain(int(task.finish[0]), int(task.finish[0])).FlattenedIntervals())

                    self.tasks_with_final_var.append(task.id)
                    logging.debug(f'Task {task.id}, new var: finish {task.finish[0]}')
                elif task.state == TaskState.InProgress:
                    # delete duration constraints
                    for phase in range(3):
                        idx = self.duration_constraints[task.id][phase].Index()
                        self.model.Proto().constraints[idx].Clear()
                    if current_time > task.start+self.task_duration[task.agent[0]][task.id][0]:
                        end_value = current_time
                    else:
                        end_value = task.start+self.task_duration[task.agent[0]][task.id][0]

                    # set end of interval to predictit from actual start time
                    self.model.Proto().variables[self.task_intervals[task.id][2].EndExpr().Index()].domain[:] = []
                    self.model.Proto().variables[self.task_intervals[task.id][2].EndExpr().Index()].domain.extend(
                        cp_model.Domain(end_value, end_value).FlattenedIntervals())


                    logging.debug(f'Task {task.id}, new var: finish {end_value}')
                else:
                    logging.warning(f'Task {task.id} state {task.state} out of If Else')

                # set start of interval to current start time of task
                self.model.Proto().variables[self.task_intervals[task.id][0].StartExpr().Index()].domain[:] = []
                self.model.Proto().variables[self.task_intervals[task.id][0].StartExpr().Index()].domain.extend(
                    cp_model.Domain(int(task.start), int(task.start)).FlattenedIntervals())

                logging.debug(f'Task {task.id}, new var: start {task.start}')

                # delete dependencies constraints
                # for dep in self.job.task_sequence:
                #     if task.id in dep.conditions:
                #         logging.debug(f'Task {dep.id} dependency {task.id}')
                #         idx = self.dependencies_constraints[dep.id][task.id].Index()
                #         self.model.Proto().constraints[idx].Clear()

            elif task.state == TaskState.AVAILABLE or task.state == TaskState.UNAVAILABLE:
                # set start of interval to current time
                self.model.Proto().variables[self.task_intervals[task.id][0].StartExpr().Index()].domain[:] = []
                self.model.Proto().variables[self.task_intervals[task.id][0].StartExpr().Index()].domain.extend(
                    cp_model.Domain(int(current_time), self.horizon).FlattenedIntervals())
            else:
                logging.warning(f'task {task.id} state {task.state} out of If Else')

    def change_agent_in_model(self, task):
        """
        Changes agent variable domain.

        :param task: Task to be redirected to another agent.
        :type task: Task
        """
        idx = self.allowedAgents[task.id].Index()
        self.model.Proto().constraints[idx].Clear()

        # task can be assignment to any agent from the task_assignment list except current agent
        self.allowedAgents[task.id] = self.model.AddAllowedAssignments(
            variables=[self.task_assignment_var[task.id]],
            tuples_list=[tuple([v]) for v in self.task_assignment[task.id] if v != self.agent_mapping[task.agent[0]]])
        
    def set_solver(self):
        # Creates the solver and solve.
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 8
        solver.parameters.random_seed = self.seed
        solver.parameters.max_time_in_seconds = 10.0
        # solver.parameters.enumerate_all_solutions = True
        solver.parameters.log_search_progress = True if logging.getLogger().level == 10 else False
        solver.parameters.search_branching = cp_model.AUTOMATIC_SEARCH
        return solver

    def set_list_of_possible_changes(self, available_tasks, agent_name, current_time, **kwargs):
        makespans = []
        for available_task in available_tasks:
            # copy model
            test_model = self.model.Clone()
            # set index of possible task and change constraint in the model
            idx = self.allowedAgents[available_task.id].Index()
            test_model.Proto().constraints[idx].Clear()
            test_model.AddAllowedAssignments(
                variables=[self.task_assignment_var[available_task.id]],
                tuples_list=[tuple([self.agent_mapping[agent_name]])])

            solver = self.set_solver()
            status = solver.Solve(test_model)
            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                makespans.append([solver.ObjectiveValue(), available_task])
                solver_info = self.get_solver_info_as_dict(solver)
                solver_info['current_time'] = current_time
                self.evaluation_run_time.append(solver_info)
            else:
                logging.error(self.model.Validate())
                self.job.__str__()

                logging.error(
                    f"Something is wrong, status {solver.StatusName(status)} and the log of the solve")
                exit(2)

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
        if self.is_there_IDLE_agent(observation_data) and self.are_there_available_tasks():
            self.refresh_variables(current_time)
            # copy model
            test_model = self.model.Clone()
            solver = self.set_solver()
            status = solver.Solve(test_model)
            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                solver_info = self.get_solver_info_as_dict(solver)
                solver_info['current_time'] = current_time
                self.evaluation_run_time.append(solver_info)
                self.schedule, self.current_makespan = self.solve(current_time=current_time)

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
            else:
                logging.error(self.job.__str__())
                logging.error(self.model.Validate())
                logging.error(f"Something is wrong, status {solver.StatusName(status)} and the log of the solve")
                raise ValueError(f"case {self.job.case}, solver_seed {self.seed}, dist_seed {self.job.seed}")
        else:
            decision = {agent_name:None for agent_name in self.job.agents}

        return decision

    def is_there_IDLE_agent(self, observation_data):
        for index, [_, agent_state, _, _] in enumerate(observation_data):
            if agent_state == AgentState.IDLE or agent_state == AgentState.DONE or agent_state == AgentState.REJECTION:
                return True
        return False

    def find_task_old(self, agent_name, agent_rejection_tasks, current_time):
        # find allocated task
        available_tasks = self.are_there_available_tasks()
        logging.debug(f'Available tasks for agent {agent_name}')
        a = [task.id for agent in available_tasks.keys() for task in available_tasks[agent]]
        if available_tasks[agent_name]:
            for task in available_tasks[agent_name]:
                if task.id not in agent_rejection_tasks:
                    if START_AVAILABLE_TASKS:
                        return available_tasks[agent_name][0]
                    else:
                        if available_tasks[agent_name][0].start <= current_time:
                            return available_tasks[agent_name][0]

        available_tasks.pop(agent_name)
        if self.are_there_available_coworker_tasks(available_tasks):
            available_tasks_id = set([task.id for agent in available_tasks.keys() for task in available_tasks[agent]])
            # if not isinstance(agent_rejection_tasks, np.ndarray):
            agent_rejection_tasks = set(agent_rejection_tasks)
            if not available_tasks_id.issubset(agent_rejection_tasks):
                self.refresh_variables(current_time)
                # copy model
                test_model = self.model.Clone()
                solver = self.set_solver()
                status = solver.Solve(test_model)
                # offset = self.get_mean_of_task_duration()/2
                if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                    solver_info = self.get_solver_info_as_dict(solver)
                    solver_info['current_time'] = current_time
                    self.evaluation_run_time.append(solver_info)
                    if solver.ObjectiveValue() < self.job.get_current_makespan():
                        self.schedule, self.current_makespan = self.solve(current_time=current_time)
                        return self.find_task(agent_name, agent_rejection_tasks, current_time)
                    else:
                        return None
                else:
                    logging.error(self.job.__str__())
                    logging.error(self.model.Validate())
                    logging.error(f"Something is wrong, status {solver.StatusName(status)} and the log of the solve")
                    raise ValueError(f"case {self.job.case}, solver_seed {self.seed}, dist_seed {self.job.seed}")
        else:
            pass
        return None

    def find_task(self, agent_name, agent_rejection_tasks, current_time):
        for task in self.schedule[agent_name]:
            if task.state == TaskState.AVAILABLE:
                if START_AVAILABLE_TASKS:
                    return task
                else:
                    if task.start <= current_time:
                        return task
        return None

    def are_there_available_tasks(self):
        for task in self.job.task_sequence:
            if task.state is TaskState.AVAILABLE:
                return True
        return False

    @staticmethod
    def are_there_available_coworker_tasks(available_tasks):
        for agent in available_tasks.keys():
            for task in available_tasks[agent]:
                if task.universal:
                    return True
        return False

    def get_mean_of_task_duration(self):
        return statistics.mean([v[0] for v in self.task_duration.values()])
