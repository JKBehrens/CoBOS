from ortools.sat.python import cp_model


def test_remove_constraints():
    model = cp_model.CpModel()

    x = model.NewIntVar(0, 10, 'x')
    y = model.NewIntVar(0, 10, 'y')
    constr = model.Add(x+y == 15)
    constr2 = model.Add(x >4)
    # constr3 = model.Add(True)

    assert len(model.Proto().constraints) == 2

    new_model = model.Clone()

    print(model.Proto().constraints)
    idx = constr.Index()
    model.Proto().constraints.remove(constr.Proto())
    assert len(model.Proto().constraints) == 1

    del new_model.Proto().constraints[idx]

    assert model.Proto() == new_model.Proto()

    constr = model.Add(True)
    del model.Proto().constraints[-1]
    # model.Proto().constraints.remove(constr.Proto())

    assert model.Proto().constraints == new_model.Proto().constraints

    # model has a new unnamed variable from the True constraint.

    model.Proto().constraints.insert(idx, constr.Proto())

    print(model.Proto().constraints)
    solver = cp_model.CpSolver()
    solver.Solve(model)
    print(solver.Value(x))


def test_clear_constraints():
    model = cp_model.CpModel()

    x = model.NewIntVar(0, 10, 'x')
    y = model.NewIntVar(0, 10, 'y')
    constr = model.Add(x+y == 15)
    constr2 = model.Add(x >4)
    constr3 = model.Add(True)

    print(model.Proto().constraints)
    idx = constr.Index()
    model.Proto().constraints[idx].Clear()
    print(model.Proto().constraints)
    idx = constr.Index()
    model.Proto().constraints[idx].Clear()
    # model.Proto().constraints.remove(constr.Proto())
    # model.Proto().constraints.insert(idx, constr3.Proto())

    print(model.Proto().constraints)
    solver = cp_model.CpSolver()
    solver.Solve(model)
    print(solver.Value(x))


def test_interval_constraints():
    model = cp_model.CpModel()

    s1 = model.NewIntVar(0, 10, 's1')
    e1 = model.NewIntVar(0, 20, 'e1')
    d1 = model.NewIntVar(0, 10, 'd1')
    duration_constrains = model.Add(d1 > 3)
    interval1 = model.NewIntervalVar(s1, d1, e1, "interval_1")

    s2 = model.NewIntVar(0, 10, 's2')
    e2 = model.NewIntVar(0, 20, 'e2')
    d2 = model.NewIntVar(0, 10, 'd2')
    model.Add(d2 > 6)
    interval2 = model.NewIntervalVar(s2, d2, e2, "interval_2")
    no_overlap = model.AddNoOverlap([interval1, interval2])
    makespan = model.NewIntVar(0, 10000, 'makespan')
    model.AddMaxEquality(makespan, [interval1.EndExpr(), interval2.EndExpr()])

    solver = set_solver()
    solver.Solve(model)
    print(f'Interval1: start {solver.Value(interval1.StartExpr())}, end {solver.Value(interval1.EndExpr())}, duration {solver.Value(interval1.SizeExpr())}')
    print(f'Interval2: start {solver.Value(interval2.StartExpr())}, end {solver.Value(interval2.EndExpr())}, duration {solver.Value(interval2.SizeExpr())}')

    idx = duration_constrains.Index()
    model.Proto().constraints[idx].Clear()
    model.Proto().variables[interval1.StartExpr().Index()].domain[:] = []
    model.Proto().variables[interval1.StartExpr().Index()].domain.extend(
        cp_model.Domain(int(1), int(1)).FlattenedIntervals())

    model.Proto().variables[interval1.EndExpr().Index()].domain[:] = []
    model.Proto().variables[interval1.EndExpr().Index()].domain.extend(
        cp_model.Domain(int(3), int(3)).FlattenedIntervals())

    solver = set_solver()
    solver.Solve(model)
    print(
        f'Interval1: start {solver.Value(interval1.StartExpr())}, end {solver.Value(interval1.EndExpr())}, duration {solver.Value(interval1.SizeExpr())}')
    print(
        f'Interval2: start {solver.Value(interval2.StartExpr())}, end {solver.Value(interval2.EndExpr())}, duration {solver.Value(interval2.SizeExpr())}')


def test_clear_constraints_in_model_clone():
    model = cp_model.CpModel()

    x = model.NewIntVar(0, 10, 'x')
    y = model.NewIntVar(0, 10, 'y')
    constr = model.Add(x + y == 15)
    constr2 = model.Add(x > 8)

    print(f'Origin model: {model.Proto().constraints}')

    clone_model = model.Clone()
    print(f'Model clone: {clone_model.Proto().constraints}')

    idx = constr.Index()
    clone_model.Proto().constraints[idx].Clear()
    clone_model.Add(x+y < 10)
    print(f'Origin model: {model.Proto().constraints}')
    print(f'Model clone: {clone_model.Proto().constraints}')

    idx = constr.Index()
    model.Proto().constraints[idx].Clear()
    constr = model.Add(x + y > 16)
    print(f'Origin model: {model.Proto().constraints}')
    print(f'Model clone: {clone_model.Proto().constraints}')
    # print(model.Proto().constraints)
    # idx = constr.Index()
    # model.Proto().constraints[idx].Clear()
    # model.Proto().constraints.remove(constr.Proto())
    # model.Proto().constraints.insert(idx, constr3.Proto())

    # print(model.Proto().constraints)
    solver = cp_model.CpSolver()
    solver.Solve(model)
    print(f'Origin model: x = {solver.Value(x)}, y = {solver.Value(y)}')
    solver = cp_model.CpSolver()
    solver.Solve(clone_model)
    print(f'Model clone: x = {solver.Value(x)}, y = {solver.Value(y)}')


def set_solver():
    # Creates the solver and solve.
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.enumerate_all_solutions = True
    solver.parameters.log_search_progress = False
    solver.parameters.search_branching = cp_model.AUTOMATIC_SEARCH
    return solver

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


