import numpy as np


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def cyclic_decay_cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0,
                                    decay_coef=0.9, warmup_decay_coef=0.8, cycle_epochs=300):
    """ 
    Create a cyclic cosine decay schedule with warmup.
    
    Args:
        base_value: the base value of the cosine decay
        final_value: the final value of the cosine decay
        epochs: the total number of epochs
        niter_per_ep: the number of iterations per epoch
        warmup_epochs: the number of warmup epochs
        start_warmup_value: the start value of the warmup
        decay_coef: the decay coefficient of peak values of the cosine decay
        warmup_decay_coef: the decay coefficient of the warmup, will affect the warmup duration of each cycle
        cycle_epochs(int or list): the number of epochs per cycle, if int, then the number of epochs per cycle is the same
        
    Returns:
        A numpy array of the schedule
    """
    
    cycle_epochs = [500,600,700]
    
    # create cosine cyclic decay schedule
    if isinstance(cycle_epochs, int):
        num_cycles = int(np.ceil(epochs / cycle_epochs))
        cycle_start_epochs = [i * cycle_epochs for i in range(num_cycles)]
        cycle_epochs = [cycle_epochs] * num_cycles
    elif isinstance(cycle_epochs, list):
        # filter cycle_epochs and remove values that are larger than epochs
        cycle_epochs = [i for i in cycle_epochs if i < epochs and i > 0]
        # add zero to the beginning of the list and epochs to the end of the list
        cycle_epochs = [0] + cycle_epochs + [epochs]
        num_cycles = len(cycle_epochs) - 1
        cycle_start_epochs = cycle_epochs

    cycle_schedule = np.array([])
    current_warmpup_epochs = warmup_epochs
    for i in range(num_cycles):
        current_cycle_start = cycle_start_epochs[i]
        current_cycle_peaks = base_value if i==0 else cycle_schedule[-1] * (1/decay_coef)
        current_cycle_epochs = cycle_epochs[i+1] - cycle_epochs[i]
        current_warmpup_epochs = warmup_epochs if i==0 else int(current_warmpup_epochs* warmup_decay_coef)
        current_cycle = cosine_scheduler(
            base_value=current_cycle_peaks,
            final_value=final_value, 
            epochs=max( int( (epochs - current_cycle_start) * 0.9), current_cycle_epochs),
            niter_per_ep=niter_per_ep, 
            warmup_epochs=current_warmpup_epochs,
            start_warmup_value=start_warmup_value if i == 0 else cycle_schedule[-1],
            )
        print(f"{i}: ", current_cycle_start, current_warmpup_epochs)
        cycle_schedule = np.concatenate((cycle_schedule, current_cycle[:current_cycle_epochs * niter_per_ep]))
        
    assert len(cycle_schedule) == epochs * niter_per_ep
    return cycle_schedule
