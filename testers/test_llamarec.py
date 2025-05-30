# tests/test_llamarec.py
import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List

# Import your modules
from prompt_generator import EnhancedPromptGenerator
from quantitative_evaluator import RecommendationEvaluator
from domain_preprocessor import EnhancedCrossDomainPreprocessor


class TestDataFixtures:
    """Test data fixtures for consistent testing"""

    @pytest.fixture
    def sample_user_history(self) -> Dict:
        """Sample user history for testing"""
        return {
            "user1": {
                "Books": {
                    "liked": [
                        {"asin": "B001", "title": "The Great Gatsby", "rating": 5.0},
                        {"asin": "B002", "title": "1984", "rating": 4.5}
                    ],
                    "disliked": [],
                    "count": 2,
                    "avg_rating": 4.75
                },
                "Movies_and_TV": {
                    "liked": [
                        {"asin": "M001", "title": "The Shawshank Redemption", "rating": 5.0}
                    ],
                    "disliked": [],
                    "count": 1,
                    "avg_rating": 5.0
                }
            },
            "user2": {
                "Books": {
                    "liked": [
                        {"asin": "B003", "title": "To Kill a Mockingbird", "rating": 4.0}
                    ],
                    "disliked": [],
                    "count": 1,
                    "avg_rating": 4.0
                }
            }
        }

    @pytest.fixture
    def sample_dataframe(self) -> pd.DataFrame:
        """Sample DataFrame for testing"""
        return pd.DataFrame({
            'reviewerID': ['user1', 'user1', 'user2', 'user2'],
            'asin': ['B001', 'B002', 'B003', 'B004'],
            'overall': [5.0, 4.5, 4.0, 3.5],
            'reviewText': ['Great book!', 'Thought-provoking', 'Classic', 'Good read'],
            'summary': ['Amazing', 'Deep', 'Timeless', 'Solid'],
            'unixReviewTime': [1500000000, 1500000001, 1500000002, 1500000003]
        })


class TestPromptGenerator(TestDataFixtures):
    """Test the enhanced prompt generator"""

    def test_prompt_generator_initialization(self):
        """Test prompt generator initializes correctly"""
        generator = EnhancedPromptGenerator()
        assert generator is not None
        assert hasattr(generator, 'templates')
        assert hasattr(generator, 'knowledge_base')

    def test_item_description_extraction(self, sample_dataframe):
        """Test item description extraction"""
        generator = EnhancedPromptGenerator()

        # Test with title column
        df_with_title = sample_dataframe.copy()
        df_with_title['title'] = ['Book A', 'Book B', 'Book C', 'Book D']

        item_map = generator.get_item_descriptions(df_with_title)

        assert len(item_map) == 4
        assert 'B001' in item_map
        assert 'Book A' in item_map['B001']

    def test_top_items_selection(self, sample_dataframe):
        """Test top items selection logic"""
        generator = EnhancedPromptGenerator()

        item_map = {
            'B001': 'The Great Gatsby',
            'B002': '1984',
            'B003': 'To Kill a Mockingbird',
            'B004': 'Animal Farm'
        }

        top_items = generator.get_top_items(
            sample_dataframe, 'user1', item_map, max_items=2
        )

        assert 'The Great Gatsby' in top_items
        assert '1984' in top_items

    def test_prompt_generation(self, sample_dataframe):
        """Test end-to-end prompt generation"""
        generator = EnhancedPromptGenerator()

        # Create source and target dataframes
        source_df = sample_dataframe[sample_dataframe['asin'].str.startswith('B')]
        target_df = sample_dataframe[sample_dataframe['asin'].str.startswith('M')]

        # If target is empty, add some mock data
        if target_df.empty:
            target_data = {
                'reviewerID': ['user1'],
                'asin': ['M001'],
                'overall': [5.0],
                'reviewText': ['Great movie!'],
                'summary': ['Amazing'],
                'unixReviewTime': [1500000000]
            }
            target_df = pd.DataFrame(target_data)

        source_map = generator.get_item_descriptions(source_df)
        target_map = generator.get_item_descriptions(target_df)

        prompts = generator.generate_prompts(
            source_df, target_df, source_map, target_map,
            'Books', 'Movies_and_TV', max_prompts=1
        )

        assert len(prompts) > 0
        assert 'input' in prompts[0]
        assert 'output' in prompts[0]
        assert 'user_id' in prompts[0]


class TestQuantitativeEvaluator(TestDataFixtures):
    """Test the quantitative evaluator"""

    def test_evaluator_initialization(self, sample_user_history):
        """Test evaluator initializes with user history"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            import json
            json.dump(sample_user_history, f)
            temp_path = f.name

        try:
            evaluator = RecommendationEvaluator(temp_path)
            assert len(evaluator.user_history) == 2
            assert 'user1' in evaluator.user_history
        finally:
            os.unlink(temp_path)

    def test_item_extraction(self):
        """Test recommendation item extraction"""
        evaluator = RecommendationEvaluator()

        # Test various response formats
        response1 = '''1. "The Matrix" - Great sci-fi action
2. "Inception" - Mind-bending thriller
3. "Interstellar" - Space epic'''

        items = evaluator.extract_recommended_items(response1, k=3)
        assert len(items) == 3
        assert "The Matrix" in items
        assert "Inception" in items
        assert "Interstellar" in items

    def test_precision_calculation(self):
        """Test precision@k calculation"""
        evaluator = RecommendationEvaluator()

        recommendations = ["Item A", "Item B", "Item C"]
        relevant_items = ["Item A", "Item C", "Item D"]

        precision = evaluator.calculate_precision_at_k(recommendations, relevant_items, k=3)

        # 2 out of 3 recommendations are relevant
        assert abs(precision - (2 / 3)) < 0.001

    def test_ndcg_calculation(self):
        """Test NDCG@k calculation"""
        evaluator = RecommendationEvaluator()

        recommendations = ["Item A", "Item B", "Item C"]
        relevant_items = ["Item A", "Item C"]

        ndcg = evaluator.calculate_ndcg_at_k(recommendations, relevant_items, k=3)

        # Should be > 0 since we have relevant items at positions 1 and 3
        assert ndcg > 0
        assert ndcg <= 1


class TestDataPreprocessor(TestDataFixtures):
    """Test the data preprocessor"""

    def test_preprocessor_initialization(self):
        """Test preprocessor initialization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = EnhancedCrossDomainPreprocessor(
                base_path=temp_dir,
                output_path=temp_dir,
                min_interactions=3,
                min_domains=2
            )

            assert processor.min_interactions == 3
            assert processor.min_domains == 2

    def test_user_filtering(self, sample_dataframe):
        """Test active user filtering"""
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = EnhancedCrossDomainPreprocessor(
                base_path=temp_dir,
                output_path=temp_dir,
                min_interactions=2,
                min_domains=2
            )

            # Mock data loading
            data = {'Books': sample_dataframe}
            filtered_data = processor.filter_active_users_and_items_enhanced(data)

            assert 'Books' in filtered_data
            # user1 has 2 interactions, user2 has 2 interactions
            # Both should pass min_interactions=2
            assert len(filtered_data['Books']) > 0


class TestIntegration(TestDataFixtures):
    """Integration tests for the full pipeline"""

    @patch('requests.post')
    def test_end_to_end_recommendation(self, mock_post, sample_user_history):
        """Test end-to-end recommendation generation"""
        # Mock Ollama response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": '''1. "The Matrix" - Great sci-fi like your book preferences
2. "Blade Runner" - Dystopian themes similar to 1984
3. "Inception" - Complex narrative structure'''
        }
        mock_post.return_value = mock_response

        # Create temporary user history file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            import json
            json.dump(sample_user_history, f)
            temp_path = f.name

        try:
            from run_enhanced_llamarec import EnhancedLlamaRecEngine

            engine = EnhancedLlamaRecEngine(
                user_history_path=temp_path,
                enable_prompt_optimization=True
            )

            result = engine.generate_enhanced_recommendation(
                user_id="user1",
                source_domain="Books",
                target_domain="Movies_and_TV",
                template_type="cross_domain_reasoning"
            )

            assert result["success"] == True
            assert "recommendations" in result
            assert "quality_score" in result
            assert result["quality_score"] > 0

        finally:
            os.unlink(temp_path)


class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_missing_user_history(self):
        """Test handling of missing user history"""
        with pytest.raises(Exception):
            RecommendationEvaluator("nonexistent_file.json")

    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames"""
        generator = EnhancedPromptGenerator()
        empty_df = pd.DataFrame()

        item_map = generator.get_item_descriptions(empty_df)
        assert len(item_map) == 0

    def test_malformed_recommendations(self):
        """Test handling of malformed recommendation text"""
        evaluator = RecommendationEvaluator()

        # Test various malformed inputs
        malformed_inputs = [
            "",  # Empty
            "No numbered list here",  # No structure
            "1. Item without quotes",  # No quotes
            "Random text with numbers 1 2 3"  # Random numbers
        ]

        for malformed_input in malformed_inputs:
            items = evaluator.extract_recommended_items(malformed_input, k=3)
            # Should handle gracefully without crashing
            assert isinstance(items, list)


class TestPerformance:
    """Performance tests for critical components"""

    def test_large_dataframe_performance(self):
        """Test performance with large DataFrames"""
        # Create large synthetic dataset
        n_rows = 10000
        large_df = pd.DataFrame({
            'reviewerID': [f'user_{i // 10}' for i in range(n_rows)],
            'asin': [f'item_{i}' for i in range(n_rows)],
            'overall': np.random.uniform(1, 5, n_rows),
            'reviewText': ['Sample review'] * n_rows,
            'unixReviewTime': range(n_rows)
        })

        generator = EnhancedPromptGenerator()

        import time
        start_time = time.time()
        item_map = generator.get_item_descriptions(large_df)
        end_time = time.time()

        # Should complete within reasonable time (adjust threshold as needed)
        assert end_time - start_time < 5.0  # 5 seconds max
        assert len(item_map) == n_rows


# Pytest configuration
@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment"""
    # Create temporary directories, mock external services, etc.
    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp()

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)

# Run with: pytest tests/ -v --cov=./ --cov-report=html