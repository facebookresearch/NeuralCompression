import numpy


def get_scheduled_params(param, param_schedule, step_counter, ignore_schedule=False):
    # e.g. schedule = dict(vals=[1., 0.1], steps=[N])
    # reduces param value by a factor of 0.1 after N steps
    if ignore_schedule is False:
        vals, steps = param_schedule['vals'], param_schedule['steps']
        assert (len(vals) == len(steps) + 1), f'Mispecified schedule! - {param_schedule}'
        idx = numpy.where(step_counter < numpy.array(steps + [step_counter + 1]))[0][0]
        param *= vals[idx]
    return param


def weighted_rate_loss(config, total_nbpp, total_qbpp, step_counter, ignore_schedule=False):
    """
    Heavily penalize the rate with weight lambda_a >> lambda_b if it exceeds
    some target r_t, otherwise penalize with lambda_b
    """
    lambda_a = get_scheduled_params(config.lambda_A, config.lambda_schedule, step_counter, ignore_schedule)
    lambda_b = get_scheduled_params(config.lambda_B, config.lambda_schedule, step_counter, ignore_schedule)

    assert lambda_a > lambda_b, "Expected lambda_a > lambda_b, got (A) {} <= (B) {}".format(lambda_a, lambda_b)

    target_bpp = get_scheduled_params(config.target_rate, config.target_schedule, step_counter, ignore_schedule)

    total_qbpp = total_qbpp.item()

    if total_qbpp > target_bpp:
        rate_penalty = lambda_a
    else:
        rate_penalty = lambda_b

    weighted_rate = rate_penalty * total_nbpp

    return weighted_rate, float(rate_penalty)
