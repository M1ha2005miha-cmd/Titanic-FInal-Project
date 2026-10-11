from torch import nn
from skorch import NeuralNetBinaryClassifier


class TitanicNet(nn.Module):
    """Полносвязная нейросеть для предсказания выживания.

    Args:
        hidden_size: число нейронов в скрытом слое.
        dropout: доля нейронов, отключаемых на каждом шаге обучения.
    """

    def __init__(self, hidden_size=16, dropout=0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LazyLinear(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
    )

    def forward(self, x):
        """Пропускает признаки через слои и возвращает логиты."""
        return self.layers(x.float())


class TitanicNetClassifier(NeuralNetBinaryClassifier):
    """Обертка skorch, приводящая целевую переменную к float.
    
    Функция потерь BCEWithLogitsLoss требует дробные метки,
    а Survived хранится как целые числа 1/0.
    """
    def get_loss(self, y_pred, y_true, *args, **kwargs):
        """Считает потерю, предварительно переведя метки в float."""
        return super().get_loss(y_pred, y_true.float(), *args, **kwargs)