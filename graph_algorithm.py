# Queston 1: Liquid Trading
class Graph:
    
    """
    This class is to create a Graph object given a list of length n, 
    then the Graph would have n vertices.
    Each Graph object will have a list of vertices which contain the Vertex object.
    """
    
    def __init__(self,lst):
        
        """
        Input: a list of length n
        Output: None
        
        This method is a constructor to construct a Graph with a list of vertices of length n, 
        then it will create a Vertex object and assign it to the correct position in the list of vertices.
        
        Example: graph([0,1,2,3]) self.vertices will contain [Vertex 0, Vertex 1, Vertex 2, Vertex 3]
        
        Time Complexity: O(N) where N is the length of the list
        Space Complexity: O(N) where N is the length of the list 
        Auxiliary Space Complexity: O(1)
        """
        
        self.vertices = [None] * len(lst)
        for i in range(len(lst)):
            self.vertices[i] = Vertex(i)
            
    def add_edges(self, lst):
        
        """
        Input: a list of list of edges exp. [[(u1,v1,w1), (u2,v2,w2)], [(u3,v3,w3), (u4,v4,w4)]]
        Output: None
        
        This method to add edges that is given in the list from the input to the correct vertices in the graph.
        
        Example: graph.add_edges([[(u1,v1,w1)]]) the method will add an edges from vertex u1 to vertex v1 with weigh w1
        
        Time Complexity: O(N) where N is the total number of edges/tuple in the list
        Space Complexity: O(N) where N is the total number of edges/tuple in the list
        Auxiliary Space Complexity: O(1)
        """
        
        for outer in lst:
            for inner in outer:
                u = inner[0]
                v = inner[1]
                w = inner[2]
                current_edge = Edge(u,v,w)
                current_vertex = self.vertices[u]
                current_vertex.add_edge(current_edge)
    
    def __str__(self):
        return_string = ""
        for vertex in self.vertices:
            return_string = return_string + "Vertex " + str(vertex) + "\n"
        return return_string
            
class Vertex:
    
    """
    This class is to create a Vertex object given a id for the Vertex.
    Each Vertex object will have an id and a list of edges that contain Edge object. 
    """
    
    def __init__(self, id):
        
        """
        Input: the name/id of the Vertex
        Output: None
        
        This method is a constructor for Vertex object with an id and a list of edges
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        Auxiliary Space Complexity: O(1)
        """
        
        self.id = id
        self.edges = []
   
    def add_edge(self, edge):
        
        """
        Input: an Edge object
        Output: None
        
        This method is to add the Edge object from the input to the list of edges of the Vertex
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        Auxiliary Space Complexity: O(1)
        """
        
        self.edges.append(edge)
        
    def __str__(self):
        return_string = str(self.id)
        for edge in self.edges:
            return_string = return_string + "\n with edges " + str(edge)
        return return_string
        
class Edge:
    
    """
    This class is to create an Edge object given the u, v and w. 
    Each edge will have a u, v and w variable. 
    u is where the start of the edge, v is where the end of the edge adn w is the weigh of the edge
    """
    
    def __init__(self, u, v, w):
        
        """
        Input: u, v, w
        Output: None
        
        This method is a constructor for Edge object
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        Auxiliary Space Complexity: O(1)
        """
        
        self.u = u
        self.v = v
        self.w = w 
        
    def __str__(self):
        return_string = str(self.u) + ", " + str(self.v) + ", " + str(self.w)
        return return_string
    
def bellman_ford(graph, prices, source, max_trades, townspeople):
    
    """
    Input: graph; prices, a list of int; starting_liquid, an int; max_trades, an int; townspeople, a list of list of tuple
    Output: the maximum value that can be obtained after performing at most max_trades trades
    
    This function will find the maximum value that can be obtained after making max_trades trade. 
    
    Time Complexity: O(TM) where T is the total number of trades available and M is the max_trades
    Space Complexity: O(N) where N is the size of the input
    Auxiliary Space Complexity: O(VM) where V is the total vertices in the graph and M is the max_trades
    """
    
    # initialize the array of value with the source price
    value = [-1] * (max_trades + 1)
    value[0] = prices[source]
    edges = []
    array = [-1] * len(graph.vertices)
    table = []
    
    # append all the edges in townspeople into the edges array
    for people in townspeople:
        for edge in people:
            edges.append(edge)
    
    # initialize the source in array to 1
    array[source] = 1
    
    # add the array to the table        
    table.append(array)
    
    # reset the array to -1
    array = [-1] * len(graph.vertices)
    
    # loop for max_trades number of times, and each iteration go through every edges in the edges array
    for i in range(1, max_trades+1):
        array[source] = 1
        for edge in edges:
            u = edge[0]
            v = edge[1]
            w = edge[2]
            new_liquid = table[i-1][u] * w      # the total litre of the new liquid v after making the trade with u
            previous_liquid = table[i-1][v]     # the previous litre of the new liquid

            # if the new capacity is larger than the previous capacity 
            # and is larger than zero and is larger than the current one
            # then make the trade 
            if (new_liquid > previous_liquid) and (new_liquid > 0) and (new_liquid > array[v]):
                array[v] = new_liquid
                current_value = array[v] * prices[v]
                
                # check if after making the trade the value is larger than the previous one 
                if current_value > value[i]:
                    value[i] = current_value
        
        # after going through every edges add the array to the table then reset the array, 
        # then go to the next iteration               
        table.append(array)
        array = [-1] * len(graph.vertices)

    # after going through total number of max_trades, get the largest value from the value array    
    max_value = value[0]
    for i in range(len(value)):
        if value[i] > max_value: 
            max_value = value[i]   
       
    return max_value

def best_trades(prices, starting_liquid, max_trades, townspeople):
    
    """
    Input: prices, a list of int; starting_liquid, an int; max_trades, an int; townspeople, a list of list of tuple
    Output: the maximum value that can be obtained after performing at most max_trades trades
    
    This method will first create a graph based on the list of prices since the length of prices is the number of liquids. 
    Then it will add edges to the graph based on the list of townspeople. 
    Then it will run bellmon_ford function to get the maximum value after performing max_trades trade
    
    Time Complexity: O(TM) where T is the total number of trades available and M is the max_trades
    Space Complexity: O(N) where N is the size of the input
    Auxiliary Space Complexity: O(1)
    """
    
    graph = Graph(prices)
    graph.add_edges(townspeople)
    return (bellman_ford(graph, prices, starting_liquid, max_trades, townspeople))


# Question 2: Optional Delivery
import math
class Opt_Delivery_Graph:
    
    """
    This class is to create a Graph object given a number n and 
    it will create a length of n of list of vertices. 
    Each Graph will have a list of vertices which contain Vertex object and 
    an integer that is the total number of vertices. 
    """
    
    def __init__(self, n):
        
        """
        Input: an integer n
        Output: None
        
        This method is a constructor to construct a Graph with a list of vertices of length n, 
        then it will create a Vertex object and assign it to the correct position in the list of vertices.
        
        Example: graph(4) self.vertices will contain [Vertex 0, Vertex 1, Vertex 2, Vertex 3]
        
        Time Complexity: O(N) where N is the integer n
        Space Complexity: O(1)
        Auxiliary Space Complexity: O(1)
        """
        
        graph_size = n
        self.no_vertices = n
        self.vertices = [None] * graph_size
        for i in range(graph_size):
            self.vertices[i] = Opt_Delivery_Vertex(i)
                
    def add_edges(self, edge):
        
        """
        Input: a list of edges exp. [(u1,v1,w1), (u2,v2,w2), (u3,v3,w3), (u4,v4,w4)]
        Output: None
        
        This method to add edges that is given in the list from the input to the correct vertices in the graph.
        Since it is a not directed graph thus, it will add two edges from u1 to v1 and from v1 to u1
        with both the same weigh w1
        
        Example: graph.add_edges([(u1,v1,w1)] the method will add an edges from vertex u1 to vertex v1 with weigh w1 and 
                 vertex v1 to vertex u1 with weigh w1 and 
                 
        Time Complexity: O(N) where N is the total number of edges/tuple in the list
        Space Complexity: O(N) where N is the total number of edges/tuple in the list
        Auxiliary Space Complexity: O(1)
        """
        
        for e in edge:
            u = e[0]
            v = e[1]
            w = e[2]
              
            # u to v
            current_edge = Opt_Delivery_Edge(u,v,w)
            current_vertex = self.vertices[u]
            current_vertex.add_edge(current_edge)
            
            # v to u
            current_edge = Opt_Delivery_Edge(v,u,w)
            current_vertex = self.vertices[v] 
            current_vertex.add_edge(current_edge)
          
    def dijkstra(self, start, end1, end2):
        
        """
        Input: start, end1 and end2 is an integer 
        Output: a list with 2 tuples first tuple is (shortest distance from start to end1, shortest route from start to end2),
                second tuple is (shortest distance from start to end2, the shortest route from start to end2)
                
        This function will find the shortest path from start to end1 and the shortest path from start to end2
        
        Time Complexity: O(E(logV)) where E is the total number of edges in the graph and V is the total number of 
                         vertices in the graph
        Space Complexity: O(1)
        Auxiliary Space Complexity: O(E) where E is the total number of edges
        """
        
        source = self.vertices[start]
        source.cost = 0 
        source.discovered = True
        heap = MinHeap(len(self.vertices))

        # add the vertex to the heap with (vertex_id, vertex_cost)
        for vertex in self.vertices:
            heap.add((vertex.id, vertex.cost))
        
        # served the first element from the min heap 
        while len(heap) > 0:
            u = self.vertices[heap.serve()[0]]
            u.visited = True
            
            # perform edge relaxation
            for edge in u.edges:
                v = self.vertices[edge.v]
                w = edge.w
                # havent discovered yet
                if v.discovered == False:
                    v.discovered = True
                    v.cost = u.cost + w
                    v.predecessor = u
                    heap.update((v.id, v.cost))

                # is in the heap but not yet finalize
                elif v.visited == False:
                    if v.cost > u.cost + w:
                        v.cost = u.cost + w
                        v.predecessor = u   
                        heap.update((v.id, v.cost)) 
        

        # get the index of the first end and second end from the array of index of vertex in heap from the min heap
        first_end = heap.index_of_vertex_in_heap[end1]
        second_end = heap.index_of_vertex_in_heap[end2]
        
        # since there's 2 cost and 2 end, put the first cost and second cost from source to end1 and end2 to a list
        total_cost = [heap.array[first_end][1], heap.array[second_end][1]]
        total_end = [end1, end2]
        
        output = []
        route = []
        final_route = []
        
        # loop for both end to backtrack the route
        for i in range(2):
            reach_start = False
            vertex = self.vertices[total_end[i]]
            
            # backtracking the route from end to start
            # loop until it has reaches the start point
            while not reach_start:
                if vertex.id == start:
                    route.append(vertex.id)
                    reach_start = True
                else:
                    route.append(vertex.id)
                    vertex = vertex.predecessor
                        
            for j in range(len(route)):
                final_route.append(route.pop())

            output.append((total_cost[i], final_route))
            route = []
            final_route = []
        
        # reset the vertices payload
        for i in self.vertices:
            i.cost = math.inf
            i.discovered = False
            i.visited = False
            i.predecessor = None
            
        return output
            
    def __str__(self):
        return_string = ""
        for vertex in self.vertices:
            return_string = return_string + "Vertex " + str(vertex) + "\n"
        return return_string
            
class Opt_Delivery_Vertex:
    
    """
    This class is to create a Vertex object given a id for the Vertex.
    Each Vertex object will have an id, a list of edges that contain Edge object, a integer cost, 
    a boolean to indicate if it's discover or not, a boolean to indicate if it's visited or not and a predecessor which is the previous Vertex
    """
    
    def __init__(self, id):
        
        """
        Input: the name/id of the Vertex
        Output: None
        
        This method is a constructor for Vertex object with an id, a list of edges, the cost, 
        a boolean to indicate if it's discover or not, a boolean to indicate if it's visited or not and a predecessor which is the previous Vertex
        discovered will be True if the Vertex have been discovered, otherwise False
        visited will be True if the Vertex have been visited, otherwise False
        predecessor will save a Vertex object which is from where the Vertex have come from 
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        Auxiliary Space Complexity: O(1)
        """
        
        self.id = id
        self.edges = []
        self.cost = math.inf
        self.discovered = False
        self.visited = False
        self.predecessor = None
   
    def add_edge(self, edge):
        
        """
        Input: an Edge object
        Output: None
        
        This method is to add the Edge object from the input to the list of edges of the Vertex
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        Auxiliary Space Complexity: O(1)
        """
        
        self.edges.append(edge)
        
    def __str__(self):
        return_string = str(self.id)
        for edge in self.edges:
            return_string = return_string + "\n with edges " + str(edge)
        return return_string
    
class Opt_Delivery_Edge:
    
    """
    This class is to create an Edge object given the u, v and w. 
    Each edge will have a u, v and w variable. 
    u is where the start of the edge, v is where the end of the edge adn w is the weigh of the edge
    """
    
    def __init__(self, u, v, w):
        
        """
        Input: u, v, w
        Output: None
        
        This method is a constructor for Edge object
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        Auxiliary Space Complexity: O(1)
        """
        
        self.u = u
        self.v = v
        self.w = w 
        
    def __str__(self):
        return_string = str(self.u) + ", " + str(self.v) + ", " + str(self.w)
        return return_string

class MinHeap:
    
    """
    This class is to create a MinHeap object and all the function that a Minheap can execute
    Each MinHeap object will have an array which is the MinHeap, a list of integer which is to indicate the position
    of a Vertex in a MinHeap and the length of the MinHeap
    
    All of the function that have implemented for MinHeap have been referred and modified from my own implementation 
    of MaxHeap from FIT1008 except for update.
    """
    
    def __init__(self, n):
        
        """
        Input: an integer n
        Output: None
        
        This is a constructor for the MinHeap object. The array is the MinHeap itself with length n where n is the number of vertex. 
        The index_of_vertex_in_heap is a list which is the index will indicate the vertex and 
        the value in the list will indicate the position of the vertex in the heap.
        
        For example: n = 5 which means there's 5 vertices, array will have a lenght of 6 since the first index is not used.
                     Then index_of_vertex_in_heap will have a length of 5. So for example if index_of_vertex_in_heap is
                     [3,5,1,2,4] means that vertex 0 is at position 3 in the heap array, vertex 1 is at position 5 in the heap array, 
                     vertex 2 is at position 1 in the heap array and so on.
                     
        Time Complexity: O(1)
        Space Complexity: O(1)
        Auxiliary Space Complexity: O(1)
        """ 
        
        self.array = [None] * (n+1)
        self.index_of_vertex_in_heap = [None] * (n)
        self.length = 0

    def __len__(self):
        
        """
        Input: None
        Output: length of the heap array
        
        This function will return the length of the heap array
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        Auxiliary Space Complexity: O(1)
        """
        
        return self.length

    def is_full(self):
        
        """
        Input: None
        Output: True if there's no space in the heap array, False otherwise
        
        This function will return a boolean to indicate if the heap array is full or not
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        Auxiliary Space Complexity: O(1)
        """
        
        return (self.length+1) == len(self.array)

    def add(self, element):
        
        """
        Input: an element which in this case a tuple with (vertex, cost)
        Output: True if there's still space in the heap array, False otherwise
        
        This function is to add the element into the heap array, after adding to the heap array, it will rise the element to 
        the correct position in the array and update the index of the vertex in the heap.
        
        Time Complexity: O(log(N)) where N is the length of the heap array
        Space Complexity: O(M) where M is the size of element
        Auxiliary Space Complexity: O(1)
        """
        
        has_space = not self.is_full()
        if has_space:
            self.length += 1
            self.array[self.length] = element
            self.index_of_vertex_in_heap[element[0]] = self.length
            self.rise(self.length)
        return has_space

    def rise(self, current):
        
        """
        Input: an integer that indicate the index of the element in the heap array that needs to be rise
        Output: None
        
        This function will rise the item in the heap array[current] to the correct position in the heap array
        
        Time Complexity: O(log(N)) where N is the length of the heap array
        Space Complexity: O(M) where M is the size of the input
        Auxiliary Space Complexity: O(1)
        """
        
        # rise until the current is the current is bigger than it's parent
        while current > 1 and self.array[current][1] < self.array[current//2][1]:
            self.swap(current, current//2)
            current = current//2

    def swap(self,i,j):
        
        """
        Input: i is the current postion, j is the new postion
        Output: None
        
        This function will swap the element on index i and j in the heap array and also update the index of vertex in the heap
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        Auxiliary Space Complexity: O(1)
        """
        
        first  = self.array[i][0]
        second = self.array[j][0]
        self.array[i], self.array[j] = self.array[j], self.array[i]
        self.index_of_vertex_in_heap[first] = j
        self.index_of_vertex_in_heap[second] = i
        
    def is_empty(self):
        
        """
        Input: None
        Output: True if the heap array is empty, False otherwise
        
        This function will return a boolean to indicate if the heap array is empty or not
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        Auxiliary Space Complexity: O(1)        
        """
        
        return self.length == 0
    
    def update(self, element):
        
        """
        Input: an element which in this case a tuple with (vertex, cost)
        Output: None
        
        This function is to update the element in the heap array, it will rise the element to 
        the correct position in the array and update the index of the vertex in the heap.
        
        Time Complexity: O(log(N)) where N is the length of the heap array
        Space Complexity: O(M) where M is the size of element
        Auxiliary Space Complexity: O(1)
        """
        
        vertex = element[0]
        new_cost = element[1]
        index_in_heap = self.index_of_vertex_in_heap[vertex]
        self.array[index_in_heap] = element
        self.rise(index_in_heap)
        
    def serve(self):
        
        """
        Input: None
        Output: the first elemen in the heap array
        
        This function will serve the first element in the heap array by swaping the first element in the heap array with the 
        last element. And it will decrease the length of the array by 1 and then update the heap to a MinHeap by sinking the new
        first element that have been swap.
        
        Time Complexity: O(log(N)) where N is the length of the heap array
        Space Complexity: O(1)
        Auxiliary Space Complexity: O(1)
        """
        
        if self.is_empty():
            raise ValueError("Heap is empty")
        item = self.array[1]
        vertex = item[0]
        self.swap(1, self.length)
        self.length -= 1
        self.sink(1)
        return item

    def sink(self, current):
        
        """
        Input: an integer that indicate the index of the element in the heap array that needs to be sink
        Output: None
        
        This function is to sink the heap array[current] to its correct position in the heap
        
        Time Complexity: O(log(N)) where N is the length of the heap array
        Space Complexity: O(M) where M is the isze of the input
        Auxiliary Space Complexity: O(1)
        """
        # sink until the it reaches the leaf or 
        # until the the current is smaller or equal to its child
        while 2*current <= self.length:
            child = self.smallest_child(current)
            if self.array[current][1] <= self.array[child][1]:
                break
            self.swap(current, child)
            current = child

    def smallest_child(self, current):
        
        """
        Input: an integer that indicate the index of the element in the heap array 
        Output: the index in the heap array of the smallest child

        This function is to find the smallest child for the item = heap_array[current]
        
        Time Complexity: O(1) 
        Space Complexity: O(M) where M is the isze of the input
        Auxiliary Space Complexity: O(1)
        """
        
        # left child smaller than right child
        # return left child
        if 2 * current == self.length or self.array[2*current][1] < self.array[2*current+1][1]:
            return 2*current
        
        # right child smaller than left child
        # return right child
        else:
            return 2*current+1

    def __str__(self):
        res =""
        for i in range(1, self.length+1):
            if i == self.length:
                res += str(self.array[i])
            else:
                res += str(self.array[i]) + ', '
        return res

def opt_delivery(n ,roads, start, end, delivery):
    
    """
    Input: n, the number of cities; roads, a list of tuples; start, an integer in range of 0 to n-1; 
           end, an integer in range of 0 to n-1; delivery, a tuple
    Output: returns a tuple (cost of travelling from start to end, a list of integer which indicate the route from start to end)
    
    This function will create a graph of n and add the edges with roads. And then it will run 3 dijkstra to find the minimum 
    cost of travelling from start to end.
    
    1st Dijkstra: from start to end and from start to pickup point
    2nd Dijkstra: from pickup point to delivery point
    3rd Dijkstra: from delivery point to end
    
    After executung the 3 Dijkstra above, the function will then decide whether to do a delivery to cut the cost of travelling 
    or not to do a delivery. If the cost of doing a delivery and the cost of not doing a delivery is the same, 
    then the function will return the routes of not doing a delivery.
    
    Time Complexity: O(Rlog(N)) where R is the total number of roads and N is the total number of cities
    Space Complexity: O(M) where M is the size of the input
    Auxiliary Space Complexity: O(R) where R is the total number of roads
    """
    
    # create a graph and add the edges to the graph
    graph = Opt_Delivery_Graph(n)
    graph.add_edges(roads)
    
    # first dijkstra run from start to end and start to pick up point
    first_dij = graph.dijkstra(start, end, delivery[0])
    no_delivery = first_dij[0]
    start_to_pickup = first_dij[1]
    
    # second dijkstra run from pick up point to deliver point
    pickup_to_deliver =  graph.dijkstra(delivery[0], delivery[1], delivery[0])[0]
    
    # third dijkstra run from deliver point to end
    deliver_to_end = graph.dijkstra(delivery[1], end, delivery[1])[0] 
    
    # the total from start to pick up to deliver to end route
    delivery_route = start_to_pickup[1] + pickup_to_deliver[1] + deliver_to_end[1]
    
    # the total cost to make a delivery minus the profit earned
    delivery_cost = (start_to_pickup[0] + pickup_to_deliver[0] + deliver_to_end[0]) - delivery[2]
    delivery_final = [delivery_route[0]]
    
    for i in range(1, len(delivery_route)):
        if delivery_route[i] != delivery_route[i-1]:
            delivery_final.append(delivery_route[i])
            
    delivery_tuple = (delivery_cost, delivery_final)     
    
    # if no delivery cost lesser than delivery than return no delivery      
    if no_delivery[0] < delivery_cost:
        return no_delivery
    
    # elif no delivery cost equal to delivery cost than return no delivery 
    elif no_delivery[0] == delivery_cost:
        return no_delivery
    
    # else if delivery cost lesser than no delivery than return delivery
    else: 
        return delivery_tuple

