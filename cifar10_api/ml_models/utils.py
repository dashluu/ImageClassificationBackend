class Loader:
    def __init__(self, train_loader, valid_loader, test_loader):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader


class ClassDist:
    def __init__(self):
        self.train_dist = dict()
        self.valid_dist = dict()
        self.test_dist = dict()


class ModelResult:
    def __init__(self):
        self.accuracy = 0.
        self.class_accuracy = dict()
        self.class_dist = ClassDist()


class TrainResult(ModelResult):
    def __init__(self):
        super().__init__()
        self.train_loss = []
        self.valid_loss = []


class TestResult(ModelResult):
    def __init__(self):
        super().__init__()
        self.test_loss = []
