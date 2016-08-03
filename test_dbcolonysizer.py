import unittest
import dbcolonysizer


class TestMeasurer(unittest.TestCase):

    def test_isupper(self):
        db = DBColonySizer()
        sizes = db.process_files(
            'example_data/images/batch1/experiment1/treatment1/replicate1/pin1/day1.JPG')
        sizes.to_csv('./tests/data1.csv')

if __name__ == '__main__':
    unittest.main()
