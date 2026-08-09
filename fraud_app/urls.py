from django.urls import path

from . import views

app_name = "fraud_app"

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("predict/", views.predict, name="predict"),
    path("evaluation/<str:reference_id>/", views.evaluation, name="evaluation"),
    path("history/", views.history, name="history"),
    path("report/<int:tx_id>/", views.transaction_report, name="transaction_report"),
    path("assistant/", views.assistant, name="assistant"),
    path("api/predict/", views.predict_api, name="predict_api"),
    path("plots/<str:name>/", views.plot_image, name="plot_image"),
]
