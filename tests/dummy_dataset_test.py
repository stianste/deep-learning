import unittest
from helpers.dummy_dataset import DummyDataset


class TestUM(unittest.TestCase):

    # TODO: Tests wont run because of stupid python imports
    def setUp(self):
        dataset = DummyDataset()
        self.data = dataset.__getitem__(0)

    def test_first(self):
        self.assertEqual(1, self.data[0][0][0])
        self.assertNotEqual(1, self.data[0][0][1])

    def test_first_edge(self):
        self.assertEqual(1, self.data[self.input_size][0][self.input_size])
        self.assertNotEqual(
            1,
            self.data[self.input_size][0][self.input_size - 1]
            )

    def test_second_edge(self):
        self.assertEqual(
            1,
            self.data[self.input_size + 1][0][self.input_size - 1]
            )
        self.assertNotEqual(
            1,
            self.data[self.input_size + 1][0][self.input_size - 1]
        )


if __name__ == '__main__':
    unittest.main()
