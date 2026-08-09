from django.db import models


class Transaction(models.Model):
    prediction = models.IntegerField()
    probability = models.FloatField()
    timestamp = models.TextField()

    class Meta:
        managed = False
        db_table = "transactions"
        ordering = ["-id"]

    def __str__(self):
        label = "Fraud" if self.prediction else "Legitimate"
        return f"{label} ({self.probability:.2%})"
