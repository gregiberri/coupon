import logging

from sklearn.metrics import accuracy_score

from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from ml.solvers.base_solver import Solver


class PytorchTabularSolver(Solver):
    def init_model(self):
        super(PytorchTabularSolver, self).init_model()

        data_config = DataConfig(
            target=['Y'],
            # target should always be a list. Multi-targets are only supported for regression. Multi-Task Classification is not implemented
            continuous_cols=list(self.val_loader.df_data[0].select_dtypes(include=['int64']).columns),
            categorical_cols=list(self.val_loader.df_data[0].select_dtypes(include=['object', 'uint8']).columns),
            validation_split=0.0)
        trainer_config = TrainerConfig(**self.config.trainer.params.dict())  # index of the GPU to use. 0, means CPU
        optimizer_config = OptimizerConfig()

        self.model = TabularModel(data_config=data_config,
                                  model_config=self.model,
                                  optimizer_config=optimizer_config,
                                  trainer_config=trainer_config)

    def load_model(self):
        try:
            self.model = TabularModel.load_from_checkpoint(self.result_dir)
        except FileNotFoundError as e:
            logging.info('No saved model found: no model is loaded.')

    def train(self):
        """
        Training the tree based networks.
        Save the model if it has better performance than the previous ones.
        """
        self.model.fit(train=self.train_loader.df,
                       validation=self.val_loader.df)
        # save model
        self.model.save_model(self.result_dir)

        self.eval()

    def eval(self):
        """
        Evaluate the model.
        """
        preds = self.model.predict(self.val_loader.df_data[0]).prediction.to_numpy()
        gt = self.val_loader.data[1]

        self.accuracy = accuracy_score(preds, gt)
        print(self.accuracy)

        if self.config.env.save_preds:
            self.save_preds(preds)
