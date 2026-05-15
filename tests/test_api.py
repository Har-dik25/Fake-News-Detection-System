"""
tests/test_api.py — Integration tests for FastAPI endpoints.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Tests for the /health endpoint (doesn't require model loading)."""

    def test_health_endpoint_exists(self):
        """Verify the health check route is registered."""
        from app import app
        routes = [r.path for r in app.routes]
        assert "/health" in routes

    def test_predict_endpoint_exists(self):
        """Verify the predict route is registered."""
        from app import app
        routes = [r.path for r in app.routes]
        assert "/predict" in routes


class TestNewsInputValidation:
    """Tests for Pydantic input validation on NewsInput."""

    def test_input_model_accepts_valid_data(self):
        from app import NewsInput
        data = NewsInput(title="Test Headline", text="This is a valid news article body.")
        assert data.title == "Test Headline"
        assert data.text == "This is a valid news article body."

    def test_input_model_rejects_empty_title(self):
        from app import NewsInput
        with pytest.raises(Exception):
            NewsInput(title="", text="Some text")

    def test_input_model_rejects_empty_text(self):
        from app import NewsInput
        with pytest.raises(Exception):
            NewsInput(title="Title", text="")

    def test_input_model_rejects_oversized_title(self):
        from app import NewsInput
        with pytest.raises(Exception):
            NewsInput(title="A" * 501, text="Some text")

    def test_input_model_rejects_oversized_text(self):
        from app import NewsInput
        with pytest.raises(Exception):
            NewsInput(title="Title", text="A" * 50001)


class TestCalibrationLogic:
    """Tests for the dynamic model calibration function."""

    def test_calibration_default_with_few_predictions(self):
        from app import detect_model_calibration, _prediction_history
        _prediction_history.clear()
        # With fewer than 10 predictions, should return default 0.85
        for i in range(5):
            _prediction_history.append(0.5)
        result = detect_model_calibration()
        assert result == 0.85

    def test_calibration_drops_when_stuck(self):
        from app import detect_model_calibration, _prediction_history
        _prediction_history.clear()
        # If model always predicts the same value, variance is ~0, trust drops
        for i in range(50):
            _prediction_history.append(0.9)
        result = detect_model_calibration()
        assert result <= 0.50

    def test_calibration_healthy_variance(self):
        from app import detect_model_calibration, _prediction_history
        _prediction_history.clear()
        # Diverse predictions = healthy variance = high trust
        import random
        random.seed(42)
        for i in range(50):
            _prediction_history.append(random.uniform(0.1, 0.9))
        result = detect_model_calibration()
        assert result >= 0.70


class TestLiveFactCheck:
    """Tests for the live_fact_check function (mocked / graceful failure)."""

    @pytest.mark.asyncio
    async def test_fact_check_returns_dict(self):
        from app import live_fact_check
        result = await live_fact_check("Test headline", "Test body text")
        assert isinstance(result, dict)
        assert "score" in result
        assert "details" in result
        assert 0.0 <= result["score"] <= 1.0
