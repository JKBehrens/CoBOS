import numpy as np
from ortools.sat.python.cp_model import CpModel, CpSolver  # type: ignore
from ortools.sat.python import cp_model  # type: ignore
from pydantic import BaseModel, validator  # type: ignore

from google.protobuf import text_format
from ortools.sat import cp_model_pb2  # type: ignore
from ortools.sat import sat_parameters_pb2  # type: ignore

# from ortools.sat.python import swig_helper


# class SolverConfig(BaseModel):
#     # params: bytes
#     max_time_in_seconds: float  # 10
#     search_branching: str  # AUTOMATIC_SEARCH
#     enumerate_all_solutions: bool  # true

#     @classmethod
#     def from_str(cls, message: str) -> "SolverConfig":
#         lines = message.split("\n")
#         key_val_arr = np.array(
#             [
#                 (line.split(":")[0], line.split(":")[1].replace(" ", ""))
#                 for line in lines
#                 if ":" in line and line.split(":")[1].__len__() > 0
#             ]
#         )
#         parsed_dict = dict(zip(key_val_arr[:, 0], key_val_arr[:, 1]))
#         return cls(**parsed_dict)


# SolverConfig(enumerate_all_solutions=" true", max_time_in_seconds="10.4", search_branching="AUTOMATIC_SEARCH")


class SolverResponseStats(BaseModel):
    status: str  # OPTIMAL
    objective: int  # 0
    best_bound: int  # 0
    integers: int  # 1
    booleans: int  # 42
    conflicts: int  # 0
    branches: int  # 42
    propagations: int  # 42
    integer_propagations: int  # 84
    restarts: int  # 42
    lp_iterations: int  # 0
    walltime: float  # 0.000574899
    usertime: float  # 0.000574933
    deterministic_time: float  #  6.64e-06
    gap_integral: float  # 0
    solution_fingerprint: int  # 0xf4751fa5d4643aaa

    @validator("solution_fingerprint", pre=True, always=True)  # type: ignore
    def set_solution_fingerprint(cls, v: int | str):
        return int(v, 16) if isinstance(v, str) else int(v)

    @classmethod
    def from_str(cls, message: str) -> "SolverResponseStats":
        lines = message.split("\n")
        key_val_arr = np.array(
            [
                line.split(":")
                for line in lines
                if ":" in line and line.split(":")[1].__len__() > 0
            ]
        )
        parsed_dict = dict(zip(key_val_arr[:, 0], key_val_arr[:, 1]))
        return cls(**parsed_dict)


class SolverRunLog(BaseModel):
    model: str
    solver_config: str
    response: str
    stats: SolverResponseStats

    def get_model(self):
        model = cp_model_pb2.CpModelProto()
        assert text_format.Parse(self.model, model)

        pmodel = CpModel()
        pmodel.Proto().CopyFrom(model)

        return pmodel

    def get_response(self) -> cp_model_pb2.CpSolverResponse:
        response_model = cp_model_pb2.CpSolverResponse()
        assert text_format.Parse(self.response, response_model)
        return response_model

    def get_solver(self):
        solver = SolverWrapper()
        params = sat_parameters_pb2.SatParameters()
        assert text_format.Parse(self.solver_config, params)
        solver.parameters.CopyFrom(params)
        return solver


class SolverWrapper(CpSolver):
    """A wrapper that logs the search process according to the settings."""

    def Solve(
        self,
        model: CpModel,
        solution_callback: cp_model.CpSolverSolutionCallback | None = None,
    ) -> cp_model_pb2.CpSolverStatus:
        solver_conf = text_format.MessageToString(self.parameters, as_utf8=True)
        model_str = text_format.MessageToString(model.Proto(), as_utf8=True)

        status = super().Solve(model=model, solution_callback=solution_callback)

        self.solver_run_log = SolverRunLog(
            model=model_str,
            solver_config=solver_conf,
            response=text_format.MessageToString(self.ResponseProto(), as_utf8=True),
            stats=SolverResponseStats.from_str(self.ResponseStats()),
        )

        return status

    def get_run_log(self) -> SolverRunLog:
        """returns a configuration that contains the solver configuration and the model to reproduce the run

        Returns:
            SolverRunLog: the log that can be used to retrieve the model and/or rerun  the solve.
        """
        try:
            return self.solver_run_log
        except AttributeError as _:
            raise RuntimeError(
                "No Log available because the solver did not run. Call Solve first."
            )

    @staticmethod
    def get_solver() -> "SolverWrapper":
        solver = SolverWrapper()
        # solver.parameters.num_search_workers = 1
        # solver.parameters.random_seed = 42
        solver.parameters.max_time_in_seconds = 10.0
        solver.parameters.enumerate_all_solutions = True
        # solver.parameters.log_search_progress = True if logging.getLogger().level == 10 else False
        solver.parameters.search_branching = cp_model.AUTOMATIC_SEARCH
        return solver
