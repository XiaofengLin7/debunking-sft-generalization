
def get_train_val_env(env_class, config: dict):

    val_env = None
    if config.env.name == 'sokoban':
        env = env_class(dim_room=(config.env.dim_x, config.env.dim_y), num_boxes=config.env.num_boxes, max_steps=config.env.max_steps, search_depth=config.env.search_depth)
    elif config.env.name == 'sokoban_reil':
        env = env_class(dim_room=(config.env.dim_x, config.env.dim_y), num_boxes=config.env.num_boxes, max_steps=config.env.max_steps, search_depth=config.env.search_depth)
    else:
        raise ValueError(f"Environment {config.env.name} not supported")

    return env, val_env