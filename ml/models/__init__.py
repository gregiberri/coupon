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
        function = getattr(xgboost, model_config.name)
        return function(**model_config.params.dict())
    else:
        raise ValueError(f'Wrong model name in model configs: {model_config.name}')
