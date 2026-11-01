import pytest

from src.config import MODELS_DIR, ROOT

pytestmark = pytest.mark.skipif(not (MODELS_DIR / "xgboost_model.json").exists(),
                                reason="run scripts/run_pipeline.py first")

PAGES = ["Overview", "Historical Air Quality", "Forecast Explorer", "Model Comparison", "Explainability", "About"]


@pytest.mark.parametrize("page", PAGES)
def test_dashboard_page_renders(page):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, at.exception
    assert at.title[0].value
