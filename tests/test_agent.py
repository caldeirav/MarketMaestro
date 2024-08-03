import unittest
from src.agent import get_stock_recommendations

class TestStockAgent(unittest.TestCase):
    def test_get_stock_recommendations(self):
        query = "What are the top 3 US tech stocks for long-term growth?"
        result = get_stock_recommendations(query)
        self.assertIn('output', result)
        self.assertTrue(len(result['output']) > 0)

if __name__ == '__main__':
    unittest.main()