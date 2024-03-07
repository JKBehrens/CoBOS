
import json
from methods.solver_wrapper import SolverRunLog, SolverWrapper, CpModel, CpSolver
from ortools.sat.python import cp_model, cp_model_helper

from google.protobuf import text_format
from ortools.sat import cp_model_pb2
from ortools.sat import sat_parameters_pb2
from ortools.sat.python import swig_helper


def test_solve_save_reload():
    solver = SolverWrapper.get_solver()
    model = CpModel()

    x: cp_model.IntVar = model.NewIntVar(0, 42, "x")

    status = solver.Solve(model=model)
    solver_conf = solver.get_run_log()
    
    assert status in [cp_model.OPTIMAL, cp_model.FEASIBLE]

    x_val: int = solver.Value(x)

    assert 0 <= x_val and x_val <= 42

    runlog_json = solver.get_run_log().json()

    run_log: SolverRunLog = SolverRunLog(**json.loads(runlog_json))

    print(run_log)

    # reproduce a search
    logged_model = run_log.get_model()
    assert logged_model.Proto() == model.Proto()

    configured_solver = run_log.get_solver()
    status = configured_solver.Solve(logged_model)

    # cheack that the solution is same
    assert solver_conf.stats.solution_fingerprint == configured_solver.get_run_log().stats.solution_fingerprint

    # start again from that model
    logged_model = run_log.get_model()
    assert logged_model.Proto() == model.Proto()

    # change the model by adding an objective
    logged_model.Maximize(logged_model.GetIntVarFromProtoIndex(0))

    configured_solver = run_log.get_solver()
    status = configured_solver.Solve(logged_model)

    # check that the solution is different
    assert not solver_conf.stats.solution_fingerprint == configured_solver.get_run_log().stats.solution_fingerprint


def test_make_model_from_string_and_back():
    """ adapted from https://github.com/google/or-tools/blob/d37317b17ca16658451cafe05085fc22c39dd6c8/ortools/sat/python/swig_helper_test.py#L93
    testing how to save and load models, configurations and responses to human readble strings.
    """
    model_string = """
      variables { domain: -10 domain: 10 }
      variables { domain: -10 domain: 10 }
      variables { domain: -461168601842738790 domain: 461168601842738790 }
      constraints {
        linear {
          vars: 0
          vars: 1
          coeffs: 1
          coeffs: 1
          domain: -9223372036854775808
          domain: 9223372036854775807
        }
      }
      constraints {
        linear {
          vars: 0
          vars: 1
          vars: 2
          coeffs: 1
          coeffs: 2
          coeffs: -1
          domain: 0
          domain: 9223372036854775807
        }
      }
      objective {
        vars: -3
        coeffs: 1
        scaling_factor: -1
      }"""
    model = cp_model_pb2.CpModelProto()
    assert text_format.Parse(model_string, model)

    model_string2 = text_format.MessageToString(model, as_utf8=True)
    model2 = cp_model_pb2.CpModelProto()
    assert text_format.Parse(model_string2, model2)
    
    assert model == model2

    parameters = sat_parameters_pb2.SatParameters()
    parameters.optimize_with_core = True

    param_str = text_format.MessageToString(parameters, as_utf8=True)
    parameters2 = sat_parameters_pb2.SatParameters()
    text_format.Parse(param_str, parameters2)
    assert parameters2 == parameters

    pmodel = CpModel()
    pmodel.Proto().CopyFrom(model)

    var0 = pmodel.GetIntVarFromProtoIndex(0)
    var0.Proto()

    solver = CpSolver()
    solver.parameters = parameters

    # solver.parameters.ParseFromString(parameters.SerializeToString())

    # standard python solve
    status = solver.Solve(pmodel)

    # solve model via swig
    solve_wrapper = swig_helper.SolveWrapper()
    solve_wrapper.set_parameters(parameters)
    solution = solve_wrapper.solve(model)

    assert cp_model_pb2.OPTIMAL == solution.status
    assert 30.0 == solution.objective_value

    assert cp_model.OPTIMAL == status
    assert 30.0 == solver.ObjectiveValue()




