"""
Working on implementing a visibility graph for more sophisticated distance to goal rewards (around obstacles)
"""

from shapely.geometry import Point, LineString, Polygon
import networkx as nx
import math


class VisabilityGraph:
    """A class to build a visibility graph for the continuous navigation environment."""

    def __init__(self, goal,obstacles,bounds,resolution= (10,10)):
        self.goal = tuple(goal)
        self.obstacles = [self.obstacle_poly(obstacle) for obstacle in obstacles]
        self.points = self.generate_discrete_points(bounds, resolution)
        self.G_static = self.build_visibility_graph_static(self.points, self.obstacles)

    def __call__(self, start):
        start = tuple(start)
        path = self.find_shortest_path_from_precomputed_graph(self.G_static, start, self.goal, self.obstacles)
        path_len = self.path_length(path)
        return path_len, path


        # self.G = build_visibility_graph(goal, self.obstacle_polys)

        # for x in np.linspace(bounds[0][0], bounds[1][0], resolution[0]):
        #     for y in np.linspace(bounds[0][1], bounds[1][1], resolution[1]):
        #         start = (x,y)
        #         G = build_visibility_graph(start,goal, self.obstacle_polys)
        #         path = nx.shortest_path(G, source=start, target=goal, weight='weight')
        #         path_len = sum(Point(path[i]).distance(Point(path[i + 1])) for i in range(len(path) - 1))
        #         # self.G.add_node(tuple(start), pt=start)


    def shortest_path(self,start, goal):
        """Find the shortest path from start to goal in the visibility graph."""
        # print(start, goal)
        start, goal = tuple(start), tuple(goal)
        # print(start, goal)
        if not hasattr(self, 'obstacle_polys'):
            self.obstacle_polys = [self.obstacle_poly(obstacle) for obstacle in self.obstacles]
        G = build_visibility_graph(start, goal, self.obstacle_polys)

        path = nx.shortest_path(G, source=start, target=goal, weight='weight')
        path_len = sum(Point(path[i]).distance(Point(path[i + 1])) for i in range(len(path) - 1))

        return path_len, path

    def obstacle_poly(self, obstacle):
        """
        Extract the vertices of a rectangle given its center, width, and height.
        """
        if obstacle['type'] == 'circle':
            return Point(obstacle['center']).buffer(obstacle['radius'])#.exterior.coords
        elif obstacle['type'] == 'rect':
            rect = obstacle
            cx, cy = rect['center']
            w, h = rect['width'], rect['height']
            corners =  [
                (cx - w / 2, cy - h / 2),
                (cx + w / 2, cy - h / 2),
                (cx + w / 2, cy + h / 2),
                (cx - w / 2, cy + h / 2)
            ]
            return Polygon(corners)#.exterior.coords

    def generate_discrete_points(self,bounds, resolution):
        """Generate grid points within the given range."""

        xs = np.linspace(bounds[0][0], bounds[1][0], resolution[0])
        ys = np.linspace(bounds[0][1], bounds[1][1], resolution[1])
        # xs = range(x_range[0], x_range[1] + 1, step)
        # ys = range(y_range[0], y_range[1] + 1, step)
        return list(product(xs, ys))

    def build_visibility_graph_static(self,discrete_points, obstacles):
        """Build a precomputed visibility graph from grid points."""
        G = nx.Graph()
        for i, p1 in enumerate(discrete_points):
            for j, p2 in enumerate(discrete_points):
                if i >= j:
                    continue
                if is_visible(p1, p2, obstacles):
                    dist = Point(p1).distance(Point(p2))
                    G.add_edge(p1, p2, weight=dist)
        return G

    def connect_point_to_graph(self,G, point, obstacles):
        """Connect a point to all visible points in the graph."""
        for node in list(G.nodes):
            if is_visible(point, node, obstacles):
                dist = Point(point).distance(Point(node))
                G.add_edge(point, node, weight=dist)

    def find_shortest_path_from_precomputed_graph(self,G_static, start, goal, obstacles):
        """Copy and use precomputed graph for pathfinding between start and goal."""
        G = G_static.copy()
        self.connect_point_to_graph(G, start, obstacles)
        self.connect_point_to_graph(G, goal, obstacles)
        return nx.shortest_path(G, source=start, target=goal, weight='weight')

    def path_length(self,path):
        return sum(Point(path[i]).distance(Point(path[i + 1])) for i in range(len(path) - 1))

    def is_visible(self,p1, p2, obstacles):
        """Check if the line between p1 and p2 intersects any obstacle."""
        line = LineString([p1, p2])
        for obs in obstacles:
            if line.crosses(obs) or line.within(obs):
                return False
        return True

