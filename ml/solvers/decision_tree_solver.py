import os
import pickle
import matplotlib.pyplot as plt
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
        self.model.fit(*self.train_loader.data)

        # save model
        with open(os.path.join(self.result_dir, 'model.pkl'), 'wb') as f:
            pickle.dump(self.model, f)
        self.save_feature_importances()

        self.eval()

    def eval(self):
        """
        Evaluate the model.
        """
        preds = self.model.predict(self.val_loader.data[0])

        if not self.phase == 'test':
            gt = self.val_loader.data[1]
            self.accuracy = accuracy_score(gt, preds)
            print(self.accuracy)
            self.save_acc()

        if self.config.env.save_preds:
            self.save_preds(preds)

    def save_feature_importances(self):
        """
        Save the feature importances bar chart to the result dir.
        """
        importance = self.model.feature_importances_

        # plot feature importance
        plt.figure(figsize=(20, 10))
        bars = plt.bar(self.val_loader.df_data[0].columns, importance)
        plt.title('feature importances')
        plt.xticks(rotation=90)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x(), yval + .002, f'{yval * 100:.1f}%')
        plt.savefig(os.path.join(self.result_dir, 'feature_importance.png'), bbox_inches="tight")
