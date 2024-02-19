from ortools.sat.python import cp_model


def test_delete_constraints():
    model = cp_model.CpModel()

    x = model.NewIntVar(0, 10, 'x')
    y = model.NewIntVar(0, 10, 'y')
    constr = model.Add(x+y == 15)
    constr2 = model.Add(x >4)
    # constr3 = model.Add(True)

    print(model.Proto().constraints)
    idx = constr.Index()
    model.Proto().constraints.remove(constr.Proto())
    constr = model.Add(True)
    model.Proto().constraints.remove(constr.Proto())
    model.Proto().constraints.insert(idx, constr.Proto())

    print(model.Proto().constraints)
    solver = cp_model.CpSolver()
    solver.Solve(model)
    print(solver.Value(x))


def test_removing_constraints():
    model = cp_model.CpModel()
    
    x = model.NewIntVar(0, 10, 'x')
    y = model.NewIntVar(0, 10, 'y')

    constr = model.AddAllowedAssignments(variables=[x], tuples_list=([8], [2]))
    
    model.Add(x+y < 10)

    # Creates the solver and solve.
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.enumerate_all_solutions = True
    solver.parameters.log_search_progress = True
    solver.parameters.search_branching = cp_model.AUTOMATIC_SEARCH

    status = solver.Solve(model)
    assert [solver.Value(x), solver.Value(y)] == [2, 0]

    # remove and add new constrain
    model.Proto().constraints.remove(constr.Proto())
    constr = model.AddAllowedAssignments(variables=[x], tuples_list=[([8])])

    status = solver.Solve(model)
    assert [solver.Value(x), solver.Value(y)] == [8, 0]

    # remove and add new constrain
    model.Proto().constraints.remove(constr.Proto())
    constr = model.AddAllowedAssignments(variables=[x, y], tuples_list=([[1, 4], [2, 6]]))

    status = solver.Solve(model)
    assert [solver.Value(x), solver.Value(y)] == [1, 4]


