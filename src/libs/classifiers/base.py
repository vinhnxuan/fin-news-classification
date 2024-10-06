import abc
    
class BaseClassifier(abc.ABC):
    """
    BaseClassifier is an abstract base class that defines the interface for all classifiers.

    Methods
    -------
    build_model(**kwargs)
        Abstract method to construct the model. Must be implemented by subclasses.
        
    train(X_train, y_train, X_eval, y_eval, **kwargs)
        Abstract method to train the classifier. Must be implemented by subclasses.
        
    evaluate(X_test, y_test)
        Abstract method to evaluate the classifier. Must be implemented by subclasses.
    """
    @abc.abstractmethod
    def build_model(self, **kwargs):
        """Construct model"""
    @abc.abstractmethod
    def train(self, X_train, y_train, X_eval, y_eval, **kwargs):
        """Train classifier"""
    @abc.abstractmethod
    def evaluate(self, X_test, y_test):
        """Evaluate classifier"""