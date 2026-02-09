import unittest
from unittest.mock import patch, MagicMock
import threading
import os
import sys

# Add parent directory so SmartCam can be imported from tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import SmartCam


class TestGetAIAnalysis(unittest.TestCase):

    @patch('SmartCam.requests.post')
    def test_successful_api_call(self, mock_post):
        """get_ai_analysis returns response text on success."""
        mock_response = MagicMock()
        mock_response.json.return_value = {'response': 'A laptop on a desk.'}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = SmartCam.get_ai_analysis(
            "base64image", "http://localhost:11434/api/generate",
            "llava-phi3", "Describe the scene.", 30
        )

        mock_post.assert_called_once_with(
            "http://localhost:11434/api/generate",
            json={
                "model": "llava-phi3",
                "prompt": "Describe the scene.",
                "images": ["base64image"],
                "stream": False
            },
            timeout=30
        )
        self.assertEqual(result, 'A laptop on a desk.')

    @patch('SmartCam.requests.post')
    def test_request_exception_returns_none(self, mock_post):
        """get_ai_analysis returns None on connection error."""
        mock_post.side_effect = SmartCam.requests.exceptions.RequestException("Connection error")
        result = SmartCam.get_ai_analysis(
            "base64image", "http://localhost:11434/api/generate",
            "llava-phi3", "Describe.", 30
        )
        self.assertIsNone(result)

    @patch('SmartCam.requests.post')
    def test_http_error_returns_none(self, mock_post):
        """get_ai_analysis returns None on HTTP error."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = SmartCam.requests.exceptions.HTTPError("404 Client Error")
        mock_post.return_value = mock_response
        result = SmartCam.get_ai_analysis(
            "base64image", "http://localhost:11434/api/generate",
            "llava-phi3", "Describe.", 30
        )
        self.assertIsNone(result)

    @patch('SmartCam.requests.post')
    def test_unexpected_error_returns_none(self, mock_post):
        """get_ai_analysis returns None on unexpected error."""
        mock_post.side_effect = ValueError("Unexpected JSON format")
        result = SmartCam.get_ai_analysis(
            "base64image", "http://localhost:11434/api/generate",
            "llava-phi3", "Describe.", 30
        )
        self.assertIsNone(result)

    @patch('SmartCam.requests.post')
    def test_missing_response_key_returns_none(self, mock_post):
        """get_ai_analysis returns None if 'response' key missing in JSON."""
        mock_response = MagicMock()
        mock_response.json.return_value = {'other_key': 'some_value'}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        result = SmartCam.get_ai_analysis(
            "base64image", "http://localhost:11434/api/generate",
            "llava-phi3", "Describe.", 30
        )
        self.assertIsNone(result)


class TestMemoryStore(unittest.TestCase):

    TMP = os.path.join(os.path.dirname(__file__), "_test_memory.json")

    def tearDown(self):
        if os.path.exists(self.TMP):
            os.remove(self.TMP)

    def test_add_and_get_context(self):
        """Stored observation appears in context."""
        store = SmartCam.MemoryStore(self.TMP)
        store.add("A laptop on a desk.", "default")
        ctx = store.get_context("default", n=1)
        self.assertIn("A laptop on a desk.", ctx)

    def test_empty_zone_returns_empty_string(self):
        """get_context returns empty string for a zone with no entries."""
        store = SmartCam.MemoryStore(self.TMP)
        self.assertEqual(store.get_context("default", n=3), "")

    def test_zone_isolation(self):
        """Observations in different zones do not bleed into each other."""
        store = SmartCam.MemoryStore(self.TMP)
        store.add("Office scene.", "office")
        store.add("Kitchen scene.", "kitchen")
        self.assertIn("Office scene.", store.get_context("office", n=1))
        self.assertNotIn("Kitchen scene.", store.get_context("office", n=1))

    def test_prunes_to_max_per_zone(self):
        """Older entries are pruned when max_per_zone is exceeded."""
        store = SmartCam.MemoryStore(self.TMP, max_per_zone=3)
        for i in range(5):
            store.add(f"Observation {i}.", "default")
        self.assertEqual(len(store._data["default"]), 3)
        self.assertEqual(store._data["default"][-1]["desc"], "Observation 4.")

    def test_persists_and_reloads(self):
        """Data survives a MemoryStore reload from disk."""
        store = SmartCam.MemoryStore(self.TMP)
        store.add("Persisted observation.", "default")
        reloaded = SmartCam.MemoryStore(self.TMP)
        self.assertIn("Persisted observation.", reloaded.get_context("default", n=1))

    def test_n_limits_context_entries(self):
        """get_context returns at most n entries."""
        store = SmartCam.MemoryStore(self.TMP)
        for i in range(10):
            store.add(f"Item {i}.", "default")
        ctx = store.get_context("default", n=3)
        self.assertEqual(ctx.count("Item"), 3)


class TestExecuteAnalysis(unittest.TestCase):

    def _make_shared_state(self):
        return {
            'latest_status': "Initializing...",
            'lock': threading.Lock()
        }

    @patch('SmartCam.get_ai_analysis')
    def test_updates_shared_state_on_success(self, mock_get_ai):
        """_execute_analysis updates latest_status with AI result."""
        mock_get_ai.return_value = "A coffee mug on a desk."
        shared_state = self._make_shared_state()
        SmartCam._execute_analysis(
            "img", "http://localhost:11434/api/generate",
            "llava-phi3", "Describe.", shared_state, 30, None, None, "default"
        )
        self.assertEqual(shared_state['latest_status'], "A coffee mug on a desk.")

    @patch('SmartCam.get_ai_analysis')
    @patch('SmartCam.speak')
    def test_does_not_speak_without_event(self, mock_speak, mock_get_ai):
        """speak is not called unless speak_once_event is set."""
        mock_get_ai.return_value = "Something on a desk."
        SmartCam._execute_analysis(
            "img", "http://localhost:11434/api/generate",
            "llava-phi3", "Describe.", self._make_shared_state(), 30, None, None, "default"
        )
        mock_speak.assert_not_called()

    @patch('SmartCam.get_ai_analysis')
    @patch('SmartCam.speak')
    def test_speaks_and_clears_event(self, mock_speak, mock_get_ai):
        """speak is called and speak_once_event is cleared when set."""
        mock_get_ai.return_value = "A laptop and mug."
        speak_event = threading.Event()
        speak_event.set()
        SmartCam._execute_analysis(
            "img", "http://localhost:11434/api/generate",
            "llava-phi3", "Describe.", self._make_shared_state(), 30,
            None, None, "default", speak_once_event=speak_event
        )
        mock_speak.assert_called_once_with("A laptop and mug.", None)
        self.assertFalse(speak_event.is_set())

    @patch('SmartCam.get_ai_analysis')
    @patch('SmartCam.speak')
    def test_failed_analysis_sets_status(self, mock_speak, mock_get_ai):
        """latest_status is set to 'Analysis failed.' when API returns None."""
        mock_get_ai.return_value = None
        shared_state = self._make_shared_state()
        SmartCam._execute_analysis(
            "img", "http://localhost:11434/api/generate",
            "llava-phi3", "Describe.", shared_state, 30, None, None, "default"
        )
        self.assertEqual(shared_state['latest_status'], "Analysis failed.")
        mock_speak.assert_not_called()

    @patch('SmartCam.get_ai_analysis')
    def test_saves_to_memory(self, mock_get_ai):
        """Successful result is saved to MemoryStore."""
        mock_get_ai.return_value = "Books on a shelf."
        tmp = os.path.join(os.path.dirname(__file__), "_test_exec_memory.json")
        try:
            memory = SmartCam.MemoryStore(tmp)
            SmartCam._execute_analysis(
                "img", "http://localhost:11434/api/generate",
                "llava-phi3", "Describe.", self._make_shared_state(), 30,
                None, memory, "default"
            )
            self.assertIn("Books on a shelf.", memory.get_context("default", n=1))
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)


if __name__ == '__main__':
    unittest.main()
