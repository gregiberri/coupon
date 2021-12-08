import os
import pickle

from sklearn.metrics import accuracy_score
from ml.solvers.base_solver import Solver


class DecisionTreeSolver(Solver):
    def load_model(self):
        self.model_filepath = os.path.join(self.result_dir, 'model.pkl')
        if os.path.exists(self.model_filepath):
            with open(self.model_filepath, 'rb') as f:
                self.model = pickle.load(f)

    def train(self):
        """
        Training the tree based networks.
        Save the model if it has better performance than the previous ones.
        """
        self.model.fit(*self.train_loader.full_data)

        # save model
        with open(os.path.join(self.result_dir, 'model.pkl'), 'wb') as f:
            pickle.dump(self.model, f)

        self.eval()

    def eval(self):
        """
        Evaluate the model.
        """
        preds = self.model.predict(self.val_loader.full_data[0])
        gt = self.val_loader.full_data[1]

        self.accuracy = accuracy_score(gt, preds)
        print(self.accuracy)

        if self.config.env.save_preds:
            self.save_preds(preds)
