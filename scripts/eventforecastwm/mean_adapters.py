"""Preserve a trained physical mean exactly while comparing residual laws."""


def attach(model,variant):
    if variant is None:return
    if variant=='posterior_v1':
        from eventforecastwm.model_posterior import attach_posterior
        attach_posterior(model)
    elif variant=='map_patch_v1':
        from eventforecastwm.model_map_patch import attach_map_patch
        attach_map_patch(model)
    elif variant=='map_posterior_v1':
        from eventforecastwm.model_map_patch import attach_map_patch
        from eventforecastwm.model_posterior import attach_posterior
        attach_map_patch(model);attach_posterior(model)
    elif variant.endswith('_v2'):
        from eventforecastwm.performance_v2.physical import attach as attach_v2
        attach_v2(model,variant)
    else:raise ValueError('Unsupported physical mean variant: '+str(variant))
    model.recipe['mean_variant']=variant


def encode(model,belief):
    variant=model.recipe.get('mean_variant')
    if variant is not None and variant.startswith('posterior_') and variant.endswith('_v2'):
        from eventforecastwm.performance_v2.physical import correct
        return correct(model,belief)
    if variant in ('posterior_v1','map_posterior_v1'):
        from eventforecastwm.model_posterior import correct_initial
        return correct_initial(model,belief)
    return belief
