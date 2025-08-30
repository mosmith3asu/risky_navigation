import numpy as np
from shapely.geometry import Point, LineString, Polygon
import networkx as nx
import math

class VisibilityGraph:
    """A class to build a visibility graph for the continuous navigation environment."""

    def __init__(self, goal, obstacles, bounds, resolution=(20, 20)):
        if np.any(np.array(resolution) < 20):
            print("Warning: visibility graph resolution is very low; consider increasing to at least (20,20) for better accuracy.")
        print('Precomuting visibility graph...', end=' ')
        self.goal = tuple(goal)
        self.obstacles = [self.create_poly(obstacle) for obstacle in obstacles]
        self.grid = self.create_grid(bounds, resolution)
        self.graph = self.build_graph(self.grid)
        self.dist_grid,self.norm_grid = self.precompute_distances(self.grid, self.graph)

        self.invalid_mask = np.where(self.norm_grid == np.inf, True,False)
        self.invalid_offset = 1e6*self.invalid_mask
        print('\t finished.')

        self.max_dist = np.max(self.norm_grid[~self.invalid_mask])  # max distance to goal in valid grid points

    def closest_idx(self, x, y):
        dx_grid = self.grid[:, :, 0] + self.invalid_offset - x
        dy_grid = self.grid[:, :, 1] + self.invalid_offset - y
        dist_grid = np.sqrt(np.power(dx_grid, 2) + np.power(dy_grid, 2))

        # Flatten and get the indices of the 4 smallest distances
        flat_indices = np.argpartition(dist_grid.ravel(), 4)[:4]

        # Convert back to 2D indices
        rc_indices = [np.unravel_index(idx, dist_grid.shape) for idx in flat_indices]

        # Sort them by actual distance so order is closest → farthest
        rc_indices = sorted(rc_indices, key=lambda rc: dist_grid[rc])

        # Get the index of the closest point to the goal
        icloses2goal = np.argmin([self.norm_grid[r, c] for r, c in rc_indices])
        r, c = rc_indices[icloses2goal]

        return r, c

    def dist(self, x, y):
        r, c = self.closest_idx(x, y)
        dist2grid = np.abs(self.grid[r, c] - np.array([x, y], dtype=float))  # ensure x,y are float for subtraction
        return self.norm_grid[r,c] + np.linalg.norm(dist2grid)

    def dist_xy(self, x, y):
        """
        Return the (dx, dy) shortest-path step from the nearest valid grid point to (x, y).
        Uses the vector field produced by precompute_distances(). If no valid grid point
        exists (unreachable/inside obstacle), returns (np.nan, np.nan).
        """
        r, c = self.closest_idx(x, y)
        dist2goal = self.dist_grid[r, c]
        dist2grid = np.abs(self.grid[r, c] - np.array([x, y], dtype=float))
        assert not np.isnan(dist2goal).any(), f"Invalid distance to goal at grid point {r, c}: {dist2goal}"
        assert not np.any(dist2goal == np.inf), f"Invalid distance to goal at grid point {r, c}: {dist2goal}"
        return dist2goal + dist2grid

    def dist_dθ(self, x, y, theta):
        """
        Convert Cartesian coordinates (x, y) to spherical coordinates (r, theta).
        r is the distance from the origin, and theta is the angle in radians.
        """
        dx,dy = self.dist_xy(x, y)
        r = np.sqrt(dx ** 2 + dy ** 2)
        theta = np.arctan2(dy, dx) - theta
        return np.array([r, theta], dtype=np.float32)

    # ----------------- helpers & builders -----------------

    def _point_in_any_obstacle(self, p):
        Pt = Point(p)
        return any(Pt.within(poly) for poly in self.obstacles)

    def _segment_clear(self, p, q):
        """
        True if the straight line from p to q does NOT intersect any obstacle.
        We also reject if either endpoint lies strictly inside an obstacle.
        Lines that merely touch an obstacle boundary are considered blocked
        (conservative; change to poly.crosses() if you want to allow tangency).
        """
        if self._point_in_any_obstacle(p) or self._point_in_any_obstacle(q):
            return False
        seg = LineString([p, q])
        for poly in self.obstacles:
            if seg.intersects(poly):
                return False
        return True

    def create_poly(self, obstacle):
        """
        Create a Shapely geometry for the obstacle.
        Accepted formats:
          - {'type':'circle','center':(x,y),'radius':r}
          - {'type':'rect','center':(cx,cy),'width':w,'height':h}
        """
        if obstacle['type'] == 'circle':
            return Point(obstacle['center']).buffer(obstacle['radius'])
        elif obstacle['type'] == 'rect':
            rect = obstacle
            cx, cy = rect['center']
            w, h = rect['width'], rect['height']
            corners = [
                (cx - w / 2, cy - h / 2),
                (cx + w / 2, cy - h / 2),
                (cx + w / 2, cy + h / 2),
                (cx - w / 2, cy + h / 2)
            ]
            return Polygon(corners)
        else:
            raise ValueError(f"Unknown obstacle type: {obstacle['type']}")

    def create_grid(self, bounds, resolution):
        """Generate grid points within the given range [min,max] x [min,max]."""
        xs = np.linspace(bounds[0][0], bounds[1][0], resolution[0])
        ys = np.linspace(bounds[1][1], bounds[0][1], resolution[1])
        grid = np.zeros([resolution[1], resolution[0], 2])  # rows=ys, cols=xs
        for c, x in enumerate(xs):
            for r, y in enumerate(ys):
                grid[r, c, :] = [x, y]
        return grid

    def build_graph(self, grid):
        """
        Build a visibility graph from all valid (non-obstacle) grid points plus the goal.
        Fully connect mutually visible pairs with Euclidean edge weights.
        """
        G = nx.Graph()

        # Collect nodes: all grid points not inside obstacles
        rows, cols, _ = grid.shape
        nodes = []
        for r in range(rows):
            for c in range(cols):
                p = tuple(grid[r, c])
                if not self._point_in_any_obstacle(p):
                    nodes.append(p)

        # Add goal node if not inside an obstacle
        if self._point_in_any_obstacle(self.goal):
            # If goal is inside an obstacle, graph is unusable for distances
            # We still return an empty graph with no goal connection.
            pass
        else:
            nodes.append(self.goal)

        # Add nodes to graph
        for p in nodes:
            G.add_node(p)

        # Connect all visible pairs. O(N^2); acceptable for moderate grids.
        # You can micro-opt by only checking visibility to k-nearest neighbors, etc.
        n = len(nodes)
        for i in range(n):
            for j in range(i + 1, n):
                p, q = nodes[i], nodes[j]
                if self._segment_clear(p, q):
                    w = math.dist(p, q)
                    G.add_edge(p, q, weight=w)

        return G

    def precompute_distances(self, grid, graph,signed=True):
        """
        Compute shortest-path vectors (dx, dy) from each grid point to the goal.
        For each valid point, returns the vector from that point to the next node
        along its shortest path to the goal.
        Returns: dx_dy_grid[rows, cols, 2], where [r,c] = (dx, dy) from point to next step.
                 np.nan where point is unreachable/invalid.
        """
        rows, cols, _ = grid.shape
        dx_dy_grid = np.full((rows, cols, 2), np.inf, dtype=float)
        norm_grid = np.full((rows, cols), np.inf, dtype=float)
        assert self.goal in graph

        for r in range(rows):
            for c in range(cols):
                path = self.shortest_path(grid[r, c], self.goal)

                if path is None:
                    dx_dy_grid[r, c] = np.inf
                    continue
                path = np.array(path)

                path_len = np.linalg.norm(path[1:] - path[:-1], axis=1)
                path_dir = path[1] - path[0] # inf first needed direction
                path_dir /= np.linalg.norm(path_dir)  # normalize direction vector
                dx, dy = np.sum(path_len) * path_dir

                dx_dy_grid[r, c] = np.array([dx, dy], dtype=float)
                norm_grid[r, c] = np.linalg.norm(np.array([dx, dy]))

        return dx_dy_grid, norm_grid



    def shortest_path(self, start, goal, graph=None):
        """
        Find the shortest path from start to goal in the visibility graph.
        start, goal are (x,y). If they're not nodes, we temporarily insert them
        (connecting to visible neighbors), compute the path, then remove them.
        """
        if graph is None:
            graph = self.graph

        # If start/goal already exist and are connected, just return path
        start = tuple(start)
        goal = tuple(goal)

        # Work on a copy so we can insert temporary nodes safely
        G = graph.copy()

        # Helper to insert a temporary node with visibility edges
        def _insert_if_needed(pt):
            if pt in G:
                return
            if self._point_in_any_obstacle(pt):
                return  # cannot insert an invalid start/goal
            G.add_node(pt)
            for q in list(G.nodes):
                if q == pt:
                    continue
                if self._segment_clear(pt, q):
                    G.add_edge(pt, q, weight=math.dist(pt, q))

        _insert_if_needed(start)
        _insert_if_needed(goal)

        try:
            path = nx.shortest_path(G, source=start, target=goal, weight='weight')
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None



