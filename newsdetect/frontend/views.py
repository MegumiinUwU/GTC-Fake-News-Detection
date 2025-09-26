from django.shortcuts import render
from django.views.decorators.http import require_POST
from django.http import JsonResponse, HttpRequest
from django.views.decorators.csrf import ensure_csrf_cookie

from ml_pipeline.infer import FakeNewsClassifier


_classifier: FakeNewsClassifier | None = None


def _get_classifier() -> FakeNewsClassifier:

    global _classifier
    if _classifier is None:
        _classifier = FakeNewsClassifier()
    return _classifier


@ensure_csrf_cookie
def chat_view(request: HttpRequest):

    return render(request, 'frontend/chat.html')


@require_POST
def predict_view(request: HttpRequest):

    title = request.POST.get('title', '')
    text = request.POST.get('text', '')
    clf = _get_classifier()
    label = clf.predict_label(title, text)
    proba = clf.predict_proba(title, text)
    return JsonResponse({
        'label': int(label),
        'label_text': 'Real' if label == 1 else 'Fake',
        'probability_real': round(proba, 4)
    })


