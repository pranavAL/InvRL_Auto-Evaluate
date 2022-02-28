def get_heuristics(MetricsInterface, last_step):
    penalty = 0
    arc_restart = MetricsInterface.getInputContainer()['Number of times user had to restart an arc'].value
    cur_score = MetricsInterface.getInputContainer()['Current path score'].value
    tot_out_path = MetricsInterface.getInputContainer()['Total time out of path'].value
    tot_pat_tim = MetricsInterface.getInputContainer()['Total path time'].value
    cur_pat_tim = MetricsInterface.getInputContainer()['Current path time'].value
    avg_tim_pat = MetricsInterface.getInputContainer()['Average time per path'].value
    avg_sco_pat = MetricsInterface.getInputContainer()['Average score per path'].value
    cur_pat_out_time = MetricsInterface.getInputContainer()['Current path time out of range'].value
    avg_pat_out_time = MetricsInterface.getInputContainer()['Average time out of path range'].value
    ball_knock = MetricsInterface.getInputContainer()['Number of tennis balls knocked over by operator'].value
    pole_touch = MetricsInterface.getInputContainer()['Number of poles touched'].value
    pole_fell = MetricsInterface.getInputContainer()['Number of poles that fell over'].value
    barr_touch = MetricsInterface.getInputContainer()['Number of barrels touches'].value
    barr_knock = MetricsInterface.getInputContainer()['Number of barrels knocked over'].value
    equip_coll = MetricsInterface.getInputContainer()['Number of equipment collisions'].value
    num_idle = MetricsInterface.getInputContainer()['Number of times machine was left idling'].value
    buck_self = MetricsInterface.getInputContainer()['Bucket Self Contact'].value
    rat_idle = MetricsInterface.getInputContainer()['Ratio of time that operator runs equipment vs idle time'].value
    coll_env = MetricsInterface.getInputContainer()['Collisions with environment'].value
    num_goal = MetricsInterface.getInputContainer()['Exercise Number of goals met'].value
    ex_time = MetricsInterface.getInputContainer()['Exercise Time'].value
    buck_truck = MetricsInterface.getInputContainer()['Safety violation bucket over truck cab'].value
    dump_truck = MetricsInterface.getInputContainer()['Safety violation dump truck contact'].value
    elec_line = MetricsInterface.getInputContainer()['Safety violation electrical lines'].value
    hum_cont = MetricsInterface.getInputContainer()['Safety violation human contact'].value
    load_hum = MetricsInterface.getInputContainer()['Safety violation load over human'].value
    park_pos = MetricsInterface.getInputContainer()['Safety violation unsafe parking position'].value
    flip_vehc = MetricsInterface.getInputContainer()['Safety violation Flipped Vehicle'].value

    new_step = np.array

    return penalty
