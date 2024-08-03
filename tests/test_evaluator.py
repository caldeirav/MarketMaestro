import unittest
from src.evaluator import evaluate_agent_response, calculate_average_scores

class TestEvaluator(unittest.TestCase):
    def test_evaluate_agent_response(self):
        query = "What are the top 3 US tech stocks for long-term growth?"
        response = "Based on current market trends and financial data, the top 3 US tech stocks for long-term growth are: 1. Company A, 2. Company B, and 3. Company C."
        evaluation = evaluate_agent_response(query, response)
        self.assertTrue(all(criterion in evaluation for criterion in ['relevance', 'specificity', 'justification', 'diversity', 'risk_awareness']))
        self.assertTrue(all('reasoning' in result and 'score' in result for result in evaluation.values()))

    def test_calculate_average_scores(self):
        results = [
            {"evaluation": {"criterion1": {"score": 8}, "criterion2": {"score": 7}}},
            {"evaluation": {"criterion1": {"score": 9}, "criterion2": {"score": 6}}}
        ]
        avg_scores = calculate_average_scores(results)
        self.assertEqual(avg_scores["criterion1"], 8.5)
        self.assertEqual(avg_scores["criterion2"], 6.5)

if __name__ == '__main__':
    unittest.main()