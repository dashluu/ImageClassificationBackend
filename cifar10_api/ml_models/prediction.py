from rest_framework import serializers


class Prediction:
    def __init__(self, label, prob_dict) -> None:
        self.label = label
        self.prob_dict = prob_dict


class PredictionSerializer(serializers.Serializer):
    label = serializers.CharField(max_length=200)
    prob_dict = serializers.DictField(child=serializers.FloatField(max_value=1, min_value=0))
