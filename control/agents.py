"""
    Agent class for control and communication with agents

    @author: Marina Ionova, student of Cybernetics and Robotics at the CTU in Prague
    @contact: marina.ionova@cvut.cz
"""
from control.agent_states import AgentState
from simulation.sim import Sim
import logging
class Agent(Sim):
    """
    Represents an agent in the simulation.

    :param name: Name of the agent.
    :type name: str
    :param tasks: List of tasks assigned to the agent.
    :type tasks: list
    """
    def __init__(self, name, job=None, **kwargs):
        super().__init__(name, job, **kwargs)
        self.name = name
        self.state = AgentState.IDLE
        self.current_task = None
        self.rejection_tasks = []
        self.delay = 0
        self.waiting = 0

    def set_start_task(self, task, start):
        """
        Sets the start time of the current task.

        :param task: Task to be started.
        :type task: Task
        :param start: Start time of the task.
        :type start: int
        """
        self.state = AgentState.PREPARATION
        self.current_task = task
        task.start = start
        task.status = 1

    def finish_task(self, time_info):
        """
        Finishes the current task.

        :param time_info: Time information of the task's phases.
        :type time_info: list
        """
        self.current_task.finish = time_info
        self.current_task.status = 2
        self.waiting = 0

    def print_current_state(self):
        """
        Returns the current state of the agent.

        :return: Current state of the agent.
        :rtype: str
        """
        if self.state == AgentState.IDLE:
            return 'waiting for task'
        else:
            return f'is doing {self.current_task.action}'

    def find_your_task(self, cl):
        """
        Finds the next available task for the agent.

        :param cl: ControlLogic class.
        :type cl: ControlLogic
        :return: Assigned task
        :rtype: Task
        """
        if not self.state == AgentState.IDLE:
            self.current_task.status = -1
            # self.refresh_task_idle()
        for i, task in enumerate(self.available_tasks):
            if task.universal and self.name == 'Human':
                if self.ask_human('execute_task', task):
                    logging.info(f'Human accept task. Task in progress...')
                    return task
                else:
                    logging.info(f'Human reject task.')
                    self.rejection_tasks.append(task.id)
                    cl.change_agent(task=task, current_agent=self)
            else:
                return task

        if not self.state == AgentState.IDLE:
            self.current_task.status = 0
        return None

    def execute_task(self, task, job, current_time, **kwargs):
        """
        Executes a task and logs the action.

        :param task: Task to be executed.
        :type task: Task
        :param job: Job to which task belongs.
        :type job: Job
        :param current_time: Current time.
        :type current_time: int
        """
        coworker = kwargs.get('coworker')
        self.current_task = task
        if task.universal and self.name == 'Human':
            if self.ask_human('execute_task', task):
                self._handle_accepted_task(task, job, current_time, coworker)
            else:
                self.state = AgentState.REJECT
                self._handle_rejected_task(task, current_time)
        else:
            self._handle_accepted_task(task, job, current_time, coworker)

    def _handle_accepted_task(self, task, job, current_time, coworker):
        coworker_task_execution = coworker.task_execution.get(coworker.name)
        self.task_execution[coworker.name] = coworker_task_execution

        self.set_start_task(task, current_time)
        self.set_task_end(self, job, current_time)
        job.in_progress_tasks.append(task.id)

        logging.info(f'{task.agent} is doing the task {task.id}. Place object {task.action["Object"]}'
                     f'to {task.action["Place"]}. TIME {current_time}')

    def _handle_rejected_task(self, task, current_time):
        logging.info(f'Human reject the task {task.id}. Place object {task.action["Object"]}'
                     f'to {task.action["Place"]}. TIME {current_time}')
        self.rejection_tasks.append(task.id)

    def get_feedback(self, job, current_time, **kwargs):
        """
        Sends feedback from an agent.

        :param job: Job to which task belongs.
        :type job: Job
        :param current_time: Current time.
        :type current_time: int
        :param coworker: Coworker
        :type coworker: Agent
        :return: Feedback from agent.
        :rtype: str
        """
        coworker = kwargs['coworker']
        self.task_execution[coworker.name] = coworker.task_execution[coworker.name]
        if self.name == 'Robot':
            state, time_info = self.get_feedback_from_robot(self.current_task, job, current_time)
            self.state = state
        else:
            state, time_info = self.check_human_task(self.current_task, job, current_time)
            self.state = state

        if self.state == AgentState.DONE:
            self.finish_task(time_info)

        return self.state, time_info

