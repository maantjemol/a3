import typing
import unittest
import numpy as np

from dynprog import DroneExtinguisher


class TestDroneExtinguisher(unittest.TestCase):
    def test_euclidean_distance(self):
        # test if distance works from one point to another, not just 0, and using infinite points
        point1 = (1, 1)
        point2 = (4, 5)
        point_inf = (np.inf, np.inf)
        dist = 5

        self.assertEqual(
            DroneExtinguisher.compute_euclidean_distance(point1, point2), dist)
        self.assertEqual(
            DroneExtinguisher.compute_euclidean_distance(point2, point1), dist)
        self.assertEqual(
            DroneExtinguisher.compute_euclidean_distance(point2, point_inf), np.inf)

    def test_compute_idle_cost(self):
        # test if an idle time of zero returns zero (isn't tested in test_usecases.py)
        de = DroneExtinguisher(forest_location=(0, 0), bags=[10, 30, 1000],
                               bag_locations=[(2.3, 1), (7, 2.7), (1, 1)], liter_cost_per_km=2,
                               liter_budget_per_day=1000, usage_cost=None)
        de.fill_travel_costs_in_liters()

        i = 0
        j = 0
        self.assertEqual(de.compute_idle_cost(0, 0, 0), 0)

    def test_dynamic_programming_with_negative_budget(self):
        forest_location = (0, 0)
        bags = [3, 9, 2, 3, 19]
        bag_locations = [(3, 4) for _ in range(len(bags))]
        liter_cost_per_km = 0.1
        liter_budget_per_day = -1  # negative budget for the day, no bags can be transported
        usage_cost = np.array([[1],
                               [1],
                               [1],
                               [1],
                               [1]])

        solution = np.inf

        de = DroneExtinguisher(
            forest_location=forest_location,
            bags=bags,
            bag_locations=bag_locations,
            liter_cost_per_km=liter_cost_per_km,
            liter_budget_per_day=liter_budget_per_day,
            usage_cost=usage_cost
        )

        de.fill_travel_costs_in_liters()
        de.dynamic_programming()
        lowest_cost = de.lowest_cost()
        self.assertEqual(lowest_cost, solution)
