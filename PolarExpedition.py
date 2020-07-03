"""
The polar expedition graph!
===========================

Contains the graph connecting the vertices (or base stations) on the map.

This is going to be the main file that you are modifying. :)

Usage:
    Contains the graph, requires the connection to vertices and edges.
"""
import math

from vertex import Vertex
from edge import Edge


# Define a "edge already exists" exception
# Don't need to modify me.
class EdgeAlreadyExists(Exception):
    """Raised when edge already exists in the graph"""

    def __init__(self, message):
        super().__init__(message)


class Graph:
    """
    Graph Class
    -----------

    Represents the graph of vertices, which is equivalent to the map of base
    stations for our polar expedition.

    Attributes:
        * vertices (list): The list of vertices
    """

    def __init__(self):
        """
        Initialises an empty graph
        """
        self._vertices = []

    def insert_vertex(self, x_pos, y_pos):
        """
        Insert the vertex storing the y_pos and x_pos

        :param x_pos: The x position of the new vertex.
        :param y_pos: The y position of the new vertex.

        :type x_pos: float
        :type y_pos: float

        :return: The new vertex, also stored in the graph.
        """

        v = Vertex(x_pos, y_pos)
        self._vertices.append(v)
        return v

    def insert_edge(self, u, v):
        """
        Inserts the edge between vertex u and v.

        We're going to assume in this assignment that all vertices given to
        this will already exist in the graph.

        :param u: Vertex U
        :param v: Vertex V

        :type u: Vertex
        :type v: Vertex

        :return: The new edge between U and V.
        """

        e = Edge(u, v)

        # Check that the edge doesn't already exist
        for i in u.edges:
            if i == e:
                # Edge already exists.
                raise EdgeAlreadyExists("Edge already exists between vertex!")

        # Add the edge to both nodes.
        u.add_edge(e)
        v.add_edge(e)

    def remove_vertex(self, v):
        """
        Removes the vertex V from the graph.
        :param v:  The pointer to the vertex to remove
        :type v: Vertex
        """

        # Remove it from the list
        del self._vertices[self._vertices.index(v)]

        # Go through and remove all edges from that node.
        while len(v.edges) != 0:
            e = v.edges.pop()
            u = self.opposite(e, v)
            u.remove_edge(e)

    @staticmethod
    def distance(u, v):
        """
        Get the distance between vertex u and v.

        :param u: A vertex to get the distance between.
        :param v: A vertex to get the distance between.

        :type u: Vertex
        :type v: Vertex
        :return: The Euclidean distance between two vertices.
        """

        # Euclidean Distance
        # sqrt( (x2-x1)^2 + (y2-y1)^2 )

        return math.sqrt(((v.x_pos - u.x_pos) ** 2) + ((v.y_pos - u.y_pos) ** 2))

    @staticmethod
    def opposite(e, v):
        """
        Returns the vertex at the other end of v.
        :param e: The edge to get the other node.
        :param v: Vertex on the edge.
        :return: Vertex at the end of the edge, or None if error.
        """

        # It must be a vertex on the edge.
        if v not in (e.u, e.v):
            return None

        if v == e.u:
            return e.v

        return e.u

    ##############################################
    # Implement the functions below
    ##############################################

    def find_emergency_range(self, v):
        maxDist = 0
        for vert in self._vertices:
            if self.distance(vert, v) > maxDist:
                maxDist = self.distance(vert, v)

        print(maxDist)

        """
        Returns the distance to the vertex W that is furthest from V.
        :param v: The vertex to start at.
        :return: The distance of the vertex W furthest away from V.
        """
        # TODO implement me!
        return maxDist

    def find_path(self, b, s, r):

        if r == 0:
            return None

        if b == s:
            singlepath = []
            singlepath.append(b)
            return singlepath

        paths = self.Djikstra(b, s, r)
        paths.pop(0)

        maxDist = 0
        for item in paths:
            if self.distance(b, item) < r:
                return self.Djikstra(b, s, r)
            else:
                return None

        """
        Find a path from vertex B to vertex S, such that the distance from B to
        every vertex in the path is within R.  If there is no path between B
        and S within R, then return None.

        :param b: Vertex B to start from.
        :param s: Vertex S to finish at.
        :param r: The maximum range of the radio.
        :return: The LIST of the VERTICES in the path.
        """
        # TODO implement me!

    def minimum_range(self, b, s):

        if b == s:
            return 0

        paths = self.Djikstra(b, s, 99999999)
        b = paths.pop(0)
        maxDist = 0
        for item in paths:
            if self.distance(item, b) >= maxDist:
                maxDist = self.distance(item, b)

        return maxDist

        """
        Returns the minimum range required to go from Vertex B to Vertex S.
        :param b: Vertex B to start from.
        :param s: Vertex S to finish at.
        :return: The minimum range in the path to go from B to S.
        """
        # TODO implement me!

    def Djikstra(self, start, end, range):
        Tracker = self.DjikstraPopulater(start)
        Tracker = self.BubbleSort(Tracker)
        while Tracker:
            someNode = Tracker.pop(0)
            adjacentVertices = self.getAllAdjacentVertices(someNode)
            for adjacentVertice in adjacentVertices:
                holder = Graph.distance(start, adjacentVertice)
                if holder < adjacentVertice.weight:
                    adjacentVertice.weight = holder
                    adjacentVertice.previous = someNode
                Tracker.sort(key=lambda vertice: vertice.weight)

        return self.DjikstraHelper(end)

    def getAllAdjacentVertices(self, node):
        adjV = []
        for edges in node.edges:
            adjV.append(self.opposite(edges, node))
        return adjV

    def BubbleSort(self, someList):

        n = len(someList)
        for i in range(n - 1):
            for j in range(0, n - i - 1):
                if someList[j].weight > someList[j + 1].weight:
                    someList[j], someList[j + 1] = someList[j + 1], someList[j]
        return someList

    def DjikstraHelper(self, end):
        shortPath = []
        pointer = end
        if pointer.previous is not None:
            while pointer is not None:
                shortPath.insert(0, pointer)
                pointer = pointer.previous
        return shortPath

    def DjikstraPopulater(self, start):
        Tracker = []

        for vertices in self._vertices:
            if vertices != start:
                vertices.weight = float("inf")
            Tracker.append(vertices)
        start.weight = 0
        return Tracker

    def move_vertex(self, v, new_x, new_y):

        for item in self._vertices:
            if item.x_pos == new_x and item.y_pos == new_y:
                break
        else:
            v.move_vertex(new_x, new_y)

        """
        Move the defined vertex.

        If there is already a vertex there, do nothing.

        :param v: The vertex to move
        :param new_x: The new X position
        :param new_y: The new Y position
        """
        # TODO implement me!


def main():
    G = Graph()

    G.insert_vertex(0, 0)
    G.insert_vertex(1, 1)
    G.insert_vertex(2, 2)

    G.insert_edge(G._vertices[0], G._vertices[1])
    G.insert_edge(G._vertices[1], G._vertices[2])

    # Find the minimum range

    print(G.find_path(G._vertices[0], G._vertices[2], 0.1231))
    # quick maths.
    expected_r = 98.02041


if __name__ == '__main__':
    main()