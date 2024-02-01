
from ortools.sat.python import cp_model


class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, variables: list[cp_model.IntVar]):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.values = []

    def on_solution_callback(self) -> None:
        for v in self.__variables:
            print(f"{v}={self.Value(v)}", end=" ")
            if v.Name() == "var1":
                self.values.append(self.Value(v))
        print()


def test_seed_solver():


    model = cp_model.CpModel()

    var1 = model.NewIntVar(10, 100, "var1")
    var2 = model.NewIntVar(0, 100, "var2")

    obj = model.NewIntVar(0, 1000, "obj")

    model.AddAbsEquality(var1 - var2, obj)

    # model.Minimize(obj)
    model.Add(obj == 0)

    solver = cp_model.CpSolver()
    solver.parameters.random_seed = 42 #self.seed
    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.num_search_workers = 1
    solver.parameters.enumerate_all_solutions = True
    solver.parameters.search_branching = cp_model.RANDOMIZED_SEARCH

    pr = VarArraySolutionPrinter([var1, var2, obj])

    status = solver.Solve(model, pr)
    val1 = pr.values



    solver = cp_model.CpSolver()
    solver.parameters.random_seed = 42 #self.seed
    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.num_search_workers = 1
    solver.parameters.enumerate_all_solutions = True
    solver.parameters.search_branching = cp_model.RANDOMIZED_SEARCH

    pr = VarArraySolutionPrinter([var1, var2, obj])

    status = solver.Solve(model, pr)
    val2 = pr.values

    assert val1 == val2

    solver = cp_model.CpSolver()
    solver.parameters.random_seed = 1 #self.seed
    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.num_search_workers = 1
    solver.parameters.enumerate_all_solutions = True
    solver.parameters.search_branching = cp_model.RANDOMIZED_SEARCH

    pr = VarArraySolutionPrinter([var1, var2, obj])

    status = solver.Solve(model, pr)
    val3 = pr.values

    assert val3 != val2


    solver = cp_model.CpSolver()
    solver.parameters.random_seed = 1 #self.seed
    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.num_search_workers = 1
    solver.parameters.enumerate_all_solutions = True
    solver.parameters.search_branching = cp_model.AUTOMATIC_SEARCH

    pr = VarArraySolutionPrinter([var1, var2, obj])

    status = solver.Solve(model, pr)
    val4 = pr.values

    assert val4 != val2

    solver = cp_model.CpSolver()
    solver.parameters.random_seed = 2 #self.seed
    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.num_search_workers = 1
    solver.parameters.enumerate_all_solutions = True
    solver.parameters.search_branching = cp_model.AUTOMATIC_SEARCH

    pr = VarArraySolutionPrinter([var1, var2, obj])

    status = solver.Solve(model, pr)
    val5 = pr.values

    assert val4 == val5

    solver = cp_model.CpSolver()
    solver.parameters.random_seed = 2 #self.seed
    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.num_search_workers = 1
    solver.parameters.enumerate_all_solutions = True
    solver.parameters.search_branching = cp_model.AUTOMATIC_SEARCH

    pr = VarArraySolutionPrinter([var1, var2, obj])

    status = solver.Solve(model, pr)
    val6 = pr.values

    assert val6 == val5


def test_seed_solver_2():


    model = cp_model.CpModel()

    var1 = model.NewIntVar(10, 100, "var1")
    var2 = model.NewIntVar(0, 100, "var2")

    obj = model.NewIntVar(0, 1000, "obj")

    model.AddAbsEquality(var1 - var2, obj)

    # model.Minimize(obj)
    # model.Add(obj == 0)

    solver = cp_model.CpSolver()
    solver.parameters.random_seed = 42 #self.seed
    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.num_search_workers = 1
    solver.parameters.enumerate_all_solutions = True
    # solver.parameters.search_branching = cp_model.RANDOMIZED_SEARCH

    pr = VarArraySolutionPrinter([var1, var2, obj])

    status = solver.Solve(model, pr)
    val1 = pr.values


    solver = cp_model.CpSolver()
    solver.parameters.random_seed = 4 #self.seed
    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.num_search_workers = 1
    solver.parameters.enumerate_all_solutions = True
    # solver.parameters.search_branching = cp_model.RANDOMIZED_SEARCH

    pr = VarArraySolutionPrinter([var1, var2, obj])

    status = solver.Solve(model, pr)
    val2 = pr.values

    assert val1 == val2


    
