


from control.jobs import Job


def test_gen_random_job():

    task_case = 1
    seed = 7
    job = Job(task_case)

    sim = generate_sim(job, seed)

    h_agent = make_agent_fake_human(job)
    r_agent = make_agent_fake_robot(job)

    scheduling_agent = make_scheduler_agent(job, h_agent, r_agent)

    agents = [scheduling_agent, h_agent, r_agent]

    state = sim.init()

    while True:
        for agent in agents:
            # the scheduling agent will reason about the actions of each agent and then inform them
            # the human and the robot act then accordingly, but possibly with failure or declining tasks
            action  = agent.choose_action(state)
            sim.apply_action(action)
            # part of the action can be to inform other agents, starting, working, canceling, and finishing tasks.
        
        # advance the world state based on the chosen actions
        state = sim.step()

    
    


