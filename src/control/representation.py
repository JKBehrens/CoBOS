import logging
import numpy as np
from pydantic import BaseModel

from control.agent_and_task_states import TaskState


class Action(BaseModel):
    object: str
    place: str


class Task(BaseModel):
    """
    Represents a task to be completed.

    :param task_description: Dictionary containing task details.
    :type task_description: dict
    """

    id: int
    action: Action
    state: TaskState | None = None
    conditions: list[int]
    universal: bool
    agent: list[str]
    start: int | None = None
    finish: list[int] | None = None

    distribution: list[tuple[tuple[int, int], tuple[int, int], tuple[float, float]]]
    rejection_prob: float

    def __str__(self):
        """
        Returns string representation of task.

        :return: String representation of task.
        :rtype: str
        """
        try:
            s1 = (
                f"ID: {self.id}, agent: {self.agent[0]}, status: {TaskState(self.state).name}, "
                + f"task action: {self.action}, conditions: {self.conditions}, universal: {self.universal}, "
                + f"start: {self.start}, finish: {self.finish}"
            )
            logging.info(s1)
        except ValueError:
            s1 = (
                f"ID: {self.id}, agent: {self.agent[0]}, state: {self.state}, "
                + f"task action: {self.action}, conditions: {self.conditions}, universal: {self.universal}, "
                + f"start: {self.start}, finish: {self.finish}"
            )
            logging.info(s1)
        return s1

    def __repr__(self) -> str:
        return self.__str__()

    def progress(self, current_time: int, duration: int) -> float:
        """
        Calculates and returns the progress of the task as a percentage.

        :param current_time: Current time.
        :type current_time: int
        :param duration: Total duration of task.
        :type duration: int
        :return: Progress of task as a percentage.
        :rtype: float
        """
        if self.start is None:
            return 0.0
        return round(
            number=(current_time - self.start / float(duration)) * 100, ndigits=2
        )

    def as_dict(self) -> dict[str, bool | int]:
        """
        Returns task details as a dictionary.

        :return: Task details as a dictionary.
        :rtype: dict
        """
        state = TaskState(self.state).name if self.state is not None else None
        return self.dict()
        return {
            "agent": copy.deepcopy(self.agent),
            "id": copy.deepcopy(self.id),
            "action": copy.deepcopy(self.action),
            "status": copy.deepcopy(state),
            "conditions": copy.deepcopy(self.conditions),
            "universal": copy.deepcopy(self.universal),
            "start": copy.deepcopy(self.start),
            "finish": copy.deepcopy(self.finish),
        }

    def get_duration(
        self,
        rand_gen: np.random.Generator | None = None,
        seed: int | None = None,
        distribution_param=None,
    ):
        if rand_gen is None:
            assert isinstance(seed, int)
            rand_gen = np.random.default_rng(seed + self.id)
        assert isinstance(rand_gen, np.random.Generator)

        if distribution_param is None:
            distribution_param = self.distribution

        duration: list[int] = []
        for phase in range(3):
            duration.append(
                self.get_phase_duration(phase, distribution_param, rand_gen, seed)
            )

        return [sum(duration), duration[0], duration[1], duration[2]]

    def get_phase_distribution(
        self,
        phase,
        distribution_param,
        rand_gen: np.random.Generator | None = None,
        seed=None,
    ):
        if rand_gen is None:
            rand_gen = np.random.default_rng(seed + self.id)

        distributions = []
        # Create distribution for each set of parameters
        for mean, scale, fail_prob in zip(*distribution_param[phase]):
            distributions.append(
                rand_gen.normal(loc=mean, scale=scale, size=int(fail_prob * 1000))
            )
        return np.concatenate(distributions)

    def get_phase_duration(
        self,
        phase: int,
        distribution_param,
        rand_gen: np.random.Generator | None = None,
        seed: int | None = None,
    ):
        if rand_gen is None:
            assert isinstance(
                seed, int
            ), "Either rand_gen or seed must be specified. None is given."
            rand_gen = np.random.default_rng(seed + self.id)
        assert isinstance(rand_gen, np.random.Generator)

        distribution = self.get_phase_distribution(
            phase, distribution_param, rand_gen, seed
        )

        # Choice a random value from the concatenated distribution
        sample = int(np.round(rand_gen.choice(distribution), decimals=0))
        if sample <= 0:
            sample = 1

        return sample


class JobDescription(BaseModel):
    tasks: list[Task]

    @staticmethod
    def from_schedule(schedule: list[dict[str, list[Task]]]) -> "JobDescription":
        tasks: list[Task] = []
        for agent, task_dict in schedule[-1].items():
            tasks.append(Task(**task_dict))

        tasks.sort(key=lambda task: task.id)

        return JobDescription(tasks=tasks)
