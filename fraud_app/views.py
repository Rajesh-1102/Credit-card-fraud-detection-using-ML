import json

from django.http import FileResponse, Http404, JsonResponse
from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .services import (
    FEATURE_ORDER,
    ProjectDependencyError,
    ASSISTANT_PROMPTS,
    get_assistant_answer,
    get_sample_choices,
    get_sample_features,
    get_plot_path,
    get_recent_transactions,
    load_dashboard_metrics,
    predict_transaction,
    get_prediction_summary,
    get_transaction_by_id,
    get_transaction_by_ref_id,
)


@login_required
def dashboard(request):
    context = {"active_page": "dashboard"}
    try:
        context["metrics"] = load_dashboard_metrics(request.GET)
        context["recent_predictions"] = get_recent_transactions(limit=5)
        context["prediction_summary"] = get_prediction_summary()
    except (FileNotFoundError, ProjectDependencyError) as exc:
        context["setup_error"] = str(exc)
    return render(request, "fraud_app/dashboard.html", context)


@login_required
@require_http_methods(["GET", "POST"])
def predict(request):
    result = None
    form_values = _default_form_values()
    error = None
    selected_sample = request.GET.get("sample") or request.POST.get("selected_sample", "")
    loaded_sample = None

    try:
        sample_choices = get_sample_choices()
        loaded_sample = get_sample_features(selected_sample)
        if loaded_sample and request.method == "GET":
            form_values.update(loaded_sample["features"])
    except (FileNotFoundError, ProjectDependencyError) as exc:
        sample_choices = []
        error = str(exc)

    if request.method == "POST":
        form_values.update({name: request.POST.get(name, 0) for name in FEATURE_ORDER})
        try:
            result = predict_transaction(request.POST)
            # If AJAX/Fetch request, return JsonResponse
            if request.headers.get("x-requested-with") == "XMLHttpRequest" or request.POST.get("ajax") == "1":
                return JsonResponse({"reference_id": result["reference_id"]})
            from django.shortcuts import redirect
            return redirect("fraud_app:evaluation", reference_id=result["reference_id"])
        except (ValueError, ProjectDependencyError, FileNotFoundError) as exc:
            if request.headers.get("x-requested-with") == "XMLHttpRequest" or request.POST.get("ajax") == "1":
                return JsonResponse({"error": str(exc)}, status=400)
            error = str(exc)

    return render(
        request,
        "fraud_app/predict.html",
        {
            "active_page": "predict",
            "feature_order": FEATURE_ORDER,
            "sample_choices": sample_choices,
            "selected_sample": selected_sample,
            "loaded_sample": loaded_sample,
            "v_feature_fields": [
                {"name": name, "value": form_values.get(name, 0)}
                for name in [f"V{i}" for i in range(1, 29)]
            ],
            "form_values": form_values,
            "result": result,
            "error": error,
        },
    )


@login_required
def history(request):
    prediction_filter = request.GET.get("prediction", "all")
    return render(
        request,
        "fraud_app/history.html",
        {
            "active_page": "history",
            "transactions": get_recent_transactions(limit=50, prediction_filter=prediction_filter),
            "prediction_filter": prediction_filter,
        },
    )


def _parse_markdown(text):
    if not text:
        return ""
    import re
    # Convert headings ### ...
    html = re.sub(r'###\s+(.*)', r'<h3 style="font-size: 1.15rem; font-weight: 700; margin-top: 15px; margin-bottom: 8px; color: var(--ink);">\1</h3>', text)
    # Convert bold **...**
    html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
    # Convert bullets * ...
    html = re.sub(r'\*\s+(.*)', r'<li style="margin-left: 15px; padding: 4px 0;">\1</li>', html)
    # Convert newlines to HTML breaks
    html = html.replace("\n", "<br>")
    return html


@login_required
def assistant(request):
    selected_question = request.POST.get("question", "") if request.method == "POST" else ""
    answer = get_assistant_answer(selected_question)

    return render(
        request,
        "fraud_app/assistant.html",
        {
            "active_page": "assistant",
            "quick_prompts": ASSISTANT_PROMPTS,
            "selected_question": selected_question,
            "assistant_answer": _parse_markdown(answer),
        },
    )


@csrf_exempt
@login_required
@require_http_methods(["POST"])
def predict_api(request):
    try:
        payload = json.loads(request.body.decode("utf-8"))
        result = predict_transaction(payload)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body."}, status=400)
    except (ValueError, ProjectDependencyError, FileNotFoundError) as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    return JsonResponse(result)


@login_required
def plot_image(request, name):
    path = get_plot_path(name)
    if not path:
        raise Http404("Plot not found.")
    return FileResponse(open(path, "rb"), content_type="image/png")


@login_required
def transaction_report(request, tx_id):
    transaction = get_transaction_by_id(tx_id)
    if not transaction:
        raise Http404("Transaction not found.")
    return render(
        request,
        "fraud_app/report.html",
        {
            "transaction": transaction,
        },
    )


@login_required
def evaluation(request, reference_id):
    transaction = get_transaction_by_ref_id(reference_id)
    if not transaction:
        raise Http404("Evaluation not found.")
    
    probability = float(transaction.get("probability", 0.0))
    prediction = int(transaction.get("prediction", 0))
    
    if prediction == 1:
        ai_explanation = (
            "The machine learning model detected multiple suspicious transaction patterns based on the anonymized dataset features. "
            "The calculated fraud probability exceeded the predefined threshold, therefore the transaction has been classified as High Risk "
            "and a simulated fraud response has been initiated."
        )
    else:
        ai_explanation = (
            "The analysed transaction follows expected behavioural patterns. "
            "The fraud probability remained below the decision threshold and the transaction has been classified as Legitimate "
            "with no immediate security action required."
        )
    
    context = {
        "transaction": transaction,
        "risk_score_pct": probability * 100.0,
        "legit_prob": 1.0 - probability,
        "confidence_score": max(probability, 1.0 - probability) * 100.0,
        "amount": transaction.get("amount", 120.00),
        "active_page": "predict",
        "ai_explanation": ai_explanation,
    }
    return render(request, "fraud_app/evaluation.html", context)


def _default_form_values():
    defaults = {name: 0 for name in FEATURE_ORDER}
    defaults["Amount"] = 120.0
    return defaults
