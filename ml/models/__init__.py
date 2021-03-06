import pytorch_tabular
import sklearn
import torchvision.models as torch_models
import xgboost


def get_model(model_config):
    """
    Select the model according to the model config name and its parameters

    :param model_config: config containing the model name as config.name and the parameters as config.params
    :return: model
    """

    if hasattr(torch_models, model_config.name):
        raise NotImplementedError

    if hasattr(sklearn.ensemble, model_config.name):
        config_params = {k: int(v) if isinstance(v, float) else v
                         for k, v in model_config.params.dict().items()}
        function = getattr(sklearn.ensemble, model_config.name)
        return function(**config_params)

    if hasattr(sklearn.tree, model_config.name):
        config_params = {k: int(v) if isinstance(v, float) else v
                         for k, v in model_config.params.dict().items()}
        function = getattr(sklearn.tree, model_config.name)
        return function(**config_params)

    if hasattr(xgboost, model_config.name):
        if isinstance(model_config.params.max_depth, float):
            model_config.params.max_depth = int(model_config.params.max_depth)
        if isinstance(model_config.params.n_estimators, float):
            model_config.params.n_estimators = int(model_config.params.n_estimators)
        function = getattr(xgboost, model_config.name)
        return function(**model_config.params.dict())

    if hasattr(pytorch_tabular.models, model_config.name):
        function = getattr(pytorch_tabular.models, model_config.name)
        return function(**model_config.params.dict())

    else:
        raise ValueError(f'Wrong model name in model configs: {model_config.name}')
