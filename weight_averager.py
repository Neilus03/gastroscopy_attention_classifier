def average_weights(model_paths):
    num_models = len(model_paths)
    avg_weights = {}

    # Initialize a list to store each model's weights
    all_weights = [torch.load(path) for path in model_paths]

    # Assume that the keys are the same across all models
    for key in all_weights[0].keys():
        avg_weights[key] = sum(w[key] for w in all_weights) / num_models

    return avg_weights
