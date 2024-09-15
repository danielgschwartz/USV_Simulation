# Created by Imran Mohammad Chowdhury and Daniel Schwartz
# Contact schwartz@cs.fsu.edu

import heapq
import numpy as np
import random
import math
import scipy.spatial
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon, Point
from math import pi, tan
from geompreds import orient2d
from matplotlib import colors
from sys import maxsize

########## Intialize key parameters, Start ##########
mapdata_file = "maze_like_mapdata_5.txt"
startNode = (0, 56)
goalNode = (50, 16)

#startNode = (0, 56)
#goalNode = (45, 45)

#startNode = (3, 14)
#goalNode = (55, 50)

min_fixed_distance = 2.0
sample_space_size = 500
nearest_neighbor_count = 16
waitTime = 10
offsetFactor = 1.0
original_sample_space_size = sample_space_size
########## Intialize key parameters, End ##########

########### Import map file and create world map, Start ##########
mapdata = open(mapdata_file, 'r').readlines()
places = []
for line in mapdata:
    temp = line[:-1]
    temp1 = temp.split(',')
    for i, j in enumerate(temp1):
        if j == ' ':
            temp1[i] = '0'
#       places.append(list(map(int, temp1)))
    places.insert(0, list(map(int, temp1)))
    # This inverts the data file in the vertical direction, making the row, column
    # indexing in the worldmap array correspond to the y and x coordinates in the
    # Cartesian system, with the lower left corner of the data file becoming the
    # (0, 0) cell in the worldmap array.
worldmap = np.array(places)
########### Import map file and create world map, End ##########

########## Generate Grassfire path, Start ##########
GF_neighbors_cords = []
GF_distance_value = []
def Grassfire(worldmap, startNode, goalNode):
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    closedlist = set()
    closed_list = {}
    distance = {goalNode: 0}
    oheap = []
    heapq.heappush(oheap, (distance[goalNode], goalNode))
    while oheap:
        current = heapq.heappop(oheap)[1]
        if current == startNode:
            path = []
            while current in closed_list:
                path.append(current)
                current = closed_list[current]
            return path
        closedlist.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            temp_distance = distance[current] + 1
            if 0 <= neighbor[0] < worldmap.shape[0]:
                if 0 <= neighbor[1] < worldmap.shape[1]:
                    # if worldmap[neighbor[1]][neighbor[0]] == 1:
                    if worldmap[neighbor[1]][neighbor[0]] == 1 or worldmap[neighbor[1]][neighbor[0]] == 2:
                        continue
                else:
                    continue
            else:
                continue
            if neighbor in closedlist:
                continue
            if neighbor not in [i[1] for i in oheap]:
                closed_list[neighbor] = current
                distance[neighbor] = temp_distance
                heapq.heappush(oheap, (distance[neighbor], neighbor))
                GF_distance_value.append(temp_distance)
                GF_neighbors_cords.append(neighbor)

def generate_GF_path(worldmap, startNode, goalNode):
    GrassfirePath = Grassfire(worldmap, startNode, goalNode)
    GrassfirePath = GrassfirePath + [goalNode]
    GrassfirePath = GrassfirePath[::-1]
    GF_xc = []
    GF_yc = []
    for i in (range(0, len(GrassfirePath))):
        x = GrassfirePath[i][0]
        y = GrassfirePath[i][1]
        GF_xc.append(x)
        GF_yc.append(y)
    GF_path = list(zip(GF_xc, GF_yc))
    return GF_path, GF_xc, GF_yc
########## Generate Grassfire path, End ##########

########## Generate Modified Grassfire path, Start ##########
MGF_neighbors_cords = []
MGF_distance_value = []
def MGrassfire(worldmap, startNode, goalNode):
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    closedlist = set()
    closed_list = {}
    distance = {goalNode: 0}
    oheap = []
    heapq.heappush(oheap, (distance[goalNode], goalNode))
    while oheap:
        current = heapq.heappop(oheap)[1]
        if current == startNode:
            path = []
            while current in closed_list:
                path.append(current)
                current = closed_list[current]
            return path
        closedlist.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            temp_distance = distance[current] + 1
            if 0 <= neighbor[0] < worldmap.shape[0]:
                if 0 <= neighbor[1] < worldmap.shape[1]:
                    # if worldmap[neighbor[1]][neighbor[0]] == 1:
                    if worldmap[neighbor[1]][neighbor[0]] == 1 or worldmap[neighbor[1]][neighbor[0]] == 2:
                        continue
                else:
                    continue
            else:
                continue

            if neighbor in closedlist:
                continue
            if neighbor not in [i[1] for i in oheap]:
                closed_list[neighbor] = current
                distance[neighbor] = temp_distance
                heapq.heappush(oheap, (distance[neighbor], neighbor))
                MGF_distance_value.append(temp_distance)
                MGF_neighbors_cords.append(neighbor)

def generate_MGF_path(worldmap, startNode, goalNode):
    MGrassfirePath = MGrassfire(worldmap, startNode, goalNode)
    MGrassfirePath = MGrassfirePath + [goalNode]
    MGrassfirePath = MGrassfirePath[::-1]
    MGF_xc = []
    MGF_yc = []
    for i in (range(0, len(MGrassfirePath))):
        x = MGrassfirePath[i][0]
        y = MGrassfirePath[i][1]
        MGF_xc.append(x)
        MGF_yc.append(y)
    MGF_path = list(zip(MGF_xc, MGF_yc))
    return MGF_path, MGF_xc, MGF_yc
########## Generate Modified Grassfire path, End ##########

########## Create fixed-obstacle and free-space k-d trees, Start ##########
obstacle = np.where((worldmap == 1) | (worldmap == 2))
freespace = np.where(worldmap == 0)
# np.where returns (row,column) coordinates for the indicated points in the array.
# These correspond to (y,x) coordinates in the Cartesian plane, as discussed above.
listOfObstacleCoordinates = list(zip(obstacle[0], obstacle[1]))
listOfFreespaceCoordinates = list(zip(freespace[0], freespace[1]))
ox = []
for i in range(len(obstacle[1])):
    ox.append(obstacle[1][i])
oy = []
for i in range(len(obstacle[0])):
    oy.append(obstacle[0][i])
# In each obstacle pair, the x coordinate is the second value, and the y coordinate is the first.
fixed_obstacle_kdtree = scipy.spatial.cKDTree(np.vstack((ox, oy)).T)

obstacle_2 = np.where(worldmap == 1)
ox_2 = []
for i in range(len(obstacle_2[1])):
    ox_2.append(obstacle_2[1][i])
oy_2 = []
for i in range(len(obstacle_2[0])):
    oy_2.append(obstacle_2[0][i])

obstacleList = list(zip(ox_2, oy_2))

fx = []
for i in range(len(freespace[1])):
    fx.append(freespace[1][i])
fy = []
for i in range(len(freespace[0])):
    fy.append(freespace[0][i])
    
# In each freespace pair, the x coordinate is the second value, and the y coordinate is the first.
fkdtree = scipy.spatial.cKDTree(np.vstack((fx, fy)).T)
########## Create fixed-obstacle and free-space k-d trees, End ##########

########## Create sample set in GF-MGF polygon, Start ##########
def search1(data, k=1):
    ind = []
    dist = []
    for i in data.T:
        idist, iind = fixed_obstacle_kdtree.query(i, k=k)
        ind.append(iind)
        dist.append(idist)
    return dist, ind

def get_random_point_in_polygon(poly):
    minx, miny, maxx, maxy = poly.bounds
    while True:
        p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        dist, ind = search1(np.array([p.x, p.y]).reshape(2, 1))
        if poly.contains(p) and dist[0] >= 0.5:
            return p

def generate_random_sampling_in_polygon(GF_path, MGF_path, GF_xc, GF_yc):
    sample_x = []
    sample_y = []
    poly = GF_path + MGF_path
    pn = Polygon(poly)
    while len(sample_x) < sample_space_size:
        point_in_poly = get_random_point_in_polygon(pn)
        tempx, tempy = np.array(point_in_poly)
        sample_x.append(tempx)
        sample_y.append(tempy)
    for i in range(len(GF_xc)):
        sample_x.append(float(GF_xc[i]))
        sample_y.append(float(GF_yc[i]))
    return sample_x, sample_y
# sample_x, sample_y are the coordinates of the sample points
########## Create sample set in GF-MGF polygon, End ##########

########## Create r-PRM road map reachability matrix, Start ##########
def collision(sx, sy, gx, gy, okdtree):
    x = sx
    y = sy
    dx = gx - sx
    dy = gy - sy
    d = math.sqrt(dx ** 2 + dy ** 2)
    yaw = math.atan2(gy - sy, gx - sx)
    if d >= 25:
        return True
    else:
        for i in range(round(d)):
            dist, ind = search1(np.array([x, y]).reshape(2, 1))
            if dist[0] <= 0.79:
                return True
            x += 1.0 * math.cos(yaw)
            y += 1.0 * math.sin(yaw)
    dist, ind = search1(np.array([gx, gy]).reshape(2, 1))
    if dist[0] <= 0.79:
        return True
    return False

def index_of_sample_point(thePoint):
    for i in range(len(sample_x)):
        x = sample_x[i]
        y = sample_y[i]
        if (x, y) == thePoint:
            return i

def index_of_open_point(thePoint):
    for i in range(len(open_points_x)):
        x = open_points_x[i]
        y = open_points_y[i]
        if (x, y) == thePoint:
            return i

#fig, ax1 = plt.subplots(figsize=(10,10))        
def drawRPRMGraph(rprm_road_map_matrix):
    plt.cla()
    global ax1
    global sample_x
    global sample_y
    cmap = colors.ListedColormap(['lightgrey', 'steelblue', 'lightgrey'])
    ax1.imshow(worldmap, cmap=cmap, origin='lower')
    ax1.scatter(startNode[0], startNode[1], marker="X", color="red", s=200)
    ax1.scatter(goalNode[0], goalNode[1], marker="X", color="blue", s=200)
    ax1.scatter(ox_2, oy_2, marker=".", color="black", s=50)
    ax1.scatter(sample_x, sample_y, marker = "x", color ="green", s=50)
    ax1.annotate("  Start", (startNode[0], startNode[1]))
    ax1.annotate("  Goal  ", (goalNode[0], goalNode[1]))
    ax1.xaxis.set_ticks(np.arange(0, 60, 1))
    plt.setp(ax1.get_xticklabels(), rotation=90, ha='center')
    ax1.yaxis.set_ticks(np.arange(0, 60, 1))
    plt.plot(GF_xc, GF_yc, color="red")
    plt.plot(MGF_xc, MGF_yc, color="black")
    for i, _ in enumerate(rprm_road_map_matrix):
        for ii in range(len(rprm_road_map_matrix[i])):
            if rprm_road_map_matrix[i][ii] != None:
                plt.plot([sample_x[i], sample_x[ii]], [sample_y[i], sample_y[ii]], "-m")
    plt.legend(['r-PRM'], loc=1)
    ax1 = plt.gca()
    leg = ax1.get_legend()
    leg.legendHandles[0].set_color('magenta')
    plt.axis("equal")
    plt.grid(True)
    plt.pause(0.01)
    plt.show()

def create_rprm_road_map_matrix(current_point, goalNode):
    global rprm_road_map_matrix
    global open_points_x
    global open_points_y
    row = index_of_sample_point(current_point)
    current_open_point_index = index_of_open_point(current_point)
    new_open_points_x = []
    new_open_points_y = []
    for i in range(len(open_points_x)):
         if i != current_open_point_index:
            new_open_points_x += [open_points_x[i]]
            new_open_points_y += [open_points_y[i]]
    open_points_x = new_open_points_x.copy()
    open_points_y = new_open_points_y.copy()
    # open_points_x.remove(open_points_x[current_open_point_index])
    # open_points_y.remove(open_points_y[current_open_point_index])
    x_1 = current_point[0]
    y_1 = current_point[1]
    open_points_kdtree = scipy.spatial.cKDTree(np.vstack((open_points_x, open_points_y)).T)
    if nearest_neighbor_count <= len(open_points_x):
        query_count = nearest_neighbor_count
    else:
        query_count = len(open_points_x)
    if query_count != 0:
        distance, index = open_points_kdtree.query(current_point, k= query_count)
        if type(index) is int:
            index = np.array([index])
        if type(distance) is float:
            distance = np.array([distance])
        for i1 in range(index.size):
            if distance[i1] == math.inf:  # if distance = math.inf, missing neighbor is ignored
                continue
            x_2 = open_points_x[index[i1]]
            y_2 = open_points_y[index[i1]]
            column = index_of_sample_point((x_2, y_2))
            if collision(x_1, y_1, x_2, y_2, fixed_obstacle_kdtree):
                rprm_road_map_matrix[row][column] = None
                rprm_road_map_matrix[column][row] = None
            else:
                rprm_road_map_matrix[row][column] = distance[i1]
                rprm_road_map_matrix[column][row] = distance[i1]
        for i1 in range(index.size):
            if distance[i1] == math.inf:  # if distance = math.inf, missing neighbor is ignored
                continue
            if index[i1] >= len(open_points_x):
                continue
            x_2 = open_points_x[index[i1]]
            y_2 = open_points_y[index[i1]]
         #   drawRPRMGraph(rprm_road_map_matrix)
            create_rprm_road_map_matrix((x_2, y_2), goalNode)
########## Create r-PRM road map reachability matrix, End ##########

########## Dijksra's Algorithm For road map reachability matrix, Start ##########
def path_to_goal(predecessors, start_node_index, goal_node_index):
    path = []
    current = goal_node_index
    path.append(goal_node_index)
    while current != start_node_index:
        path.append(predecessors[current])
        current = predecessors[current]
    path.reverse()
    return path

def dijkstra(start_node_index, goal_node_index, vertices, weights):
    shortest_path_estimate = [math.inf] * vertices
    shortest_path_estimate[start_node_index] = 0
    predecessors = [None] * vertices
    the_queue = []
    unvisited_vertices = set()
    for vertex in range(vertices):
        unvisited_vertices.add(vertex)
    for vertex in range(vertices):
        heapq.heappush(the_queue, (shortest_path_estimate[vertex], vertex))
    while the_queue:
        u = heapq.heappop(the_queue)[1]
        unvisited_vertices.remove(u)
        for v in range(vertices):
            if weights[u][v] is not None and v in unvisited_vertices:
                if shortest_path_estimate[v] > shortest_path_estimate[u] + weights[u][v]:
                    shortest_path_estimate[v] = shortest_path_estimate[u] + weights[u][v]
                    queue_temp = the_queue
                    the_queue = []
                    for (x1, y1) in queue_temp:
                        if y1 == v:
                            heapq.heappush(the_queue, (shortest_path_estimate[v], y1))
                        else:
                            heapq.heappush(the_queue, (x1, y1))
                    predecessors[v] = u

    return path_to_goal(predecessors, start_node_index, goal_node_index)
########## Dijksra's algorithm for road map reachability matrix, end ##########

########## Run Dijkstra's algorithm on the r-prm matrix, Start ##########
def generate_prm_mgf_path(sample_space_size, GF_xc, rprm_road_map_matrix, sample_x, sample_y):
    start_point = sample_space_size + len(GF_xc) - 1
    goal_point = sample_space_size
    the_prm_path = dijkstra(start_point, goal_point, sample_space_size + len(GF_xc), rprm_road_map_matrix)
    prm_mgf_x = []
    prm_mgf_y = []
    for i in range(len(the_prm_path)):
        prm_mgf_x.append(sample_x[the_prm_path[i]])
        prm_mgf_y.append(sample_y[the_prm_path[i]])
    return list(zip(prm_mgf_x, prm_mgf_y))
########## Run Dijkstra's algorithm on the r-prm matrix, End ##########

########## Create global path as a list of x,y pairs that are 1 meter apart, Start ##########
def get_line_segment_waypoints(x1, y1, x2, y2):
    # (x1, y1) and (x2, y2) are the end points.
    # It is assumed that each grid step in the x or y direction is 20 meters,
    # so the overall display is 580 m by 580 m.  Waypoints are generated along the line
    # segment at 1 meter intervals, with the exception that the last interval may be
    # shorter than 1 meter, which happens if the line segment length is not an integer
    # (and it usually isn't).
    x1 = x1 * 20
    y1 = y1 * 20
    x2 = x2 * 20
    y2 = y2 * 20
    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    sine_angle = (y2 - y1) / length
    cosine_angle = (x2 - x1) / length
    line_segment_waypoints = []
    line_segment_waypoints.append((x1, y1))
    for i in range(1, int(length)):
        if i % 8 == 0:
            # selects every eighth point as a waypoint thereby multiplying the speed of the uuv by eight
            line_segment_waypoints.append((x1 + i * cosine_angle, y1 + i * sine_angle))
    line_segment_waypoints.append((x2, y2))
    return line_segment_waypoints

def generate_global_path(PRM_MGF_path):
    the_global_path = []
    for i in range(len(PRM_MGF_path) - 1) :
        x1 = PRM_MGF_path[i][0]
        y1 = PRM_MGF_path[i][1]
        x2 = PRM_MGF_path[i + 1][0]
        y2 = PRM_MGF_path[i + 1][1]
        line_segment_waypoints = get_line_segment_waypoints(x1, y1, x2, y2)
        length = len(the_global_path)
        if length > 0 and the_global_path[length - 1] == line_segment_waypoints[0]:
            # don't allow duplicate waypoints
            the_global_path = the_global_path[:-1] + line_segment_waypoints
        else:
            the_global_path = the_global_path + line_segment_waypoints
    for i in range(1, len(the_global_path)):
        if the_global_path[i] == the_global_path[i-1]:
            print("duplicate point = ", i)
    return the_global_path
########## Create global path as a list of x.y pairs that are 1 meter apart, End ##########

########## Create global path from Some Node to Goal, Start ##########
open_points_x = []
open_points_y = []
rprm_road_map_matrix = []
GF_xc = []
GF_yc = []
MGF_xc = []
MGF_yc = []
PRM_MGF_path = []
sample_x = []
sample_y = []
def create_new_global_path(startNode, goalNode):
    global GF_xc, GF_yc, MGF_xc, MGF_yc
    GF_path, GF_xc, GF_yc = generate_GF_path(worldmap, startNode, goalNode)
    MGF_path, MGF_xc, MGF_yc = generate_MGF_path(worldmap, startNode, goalNode)
    global sample_x, sample_y
    sample_x, sample_y = generate_random_sampling_in_polygon(GF_path, MGF_path, GF_xc, GF_yc)
    global open_points_x, open_points_y
    open_points_x = sample_x.copy()
    open_points_y = sample_y.copy()
    global rprm_road_map_matrix
    rprm_road_map_matrix = np.full((sample_space_size + len(GF_xc), sample_space_size + len(GF_xc)), None)
    create_rprm_road_map_matrix(startNode, goalNode)
    global PRM_MGF_path
    PRM_MGF_path = generate_prm_mgf_path(sample_space_size, GF_xc, rprm_road_map_matrix, sample_x, sample_y)
    return generate_global_path(PRM_MGF_path)
global_path = create_new_global_path(startNode, goalNode)
########## ########## Create global path from Some Node to Goal, End ##########

######### Compute length of PRM_MGF path, Start ##########
def length_prm_mgf_path():
    length_prm_mgf = 0
    for i in (range(0, len(PRM_MGF_path) - 1)):
        length_prm_mgf = length_prm_mgf + math.sqrt(
            (PRM_MGF_path[i + 1][0] - PRM_MGF_path[i][0]) ** 2 + (PRM_MGF_path[i + 1][1] - PRM_MGF_path[i][1]) ** 2)
    return length_prm_mgf
length_original_prm_mgf_path = length_prm_mgf_path()
########## Compute length of PRM_MGF path, End ##########
#shows the GF path
fig, ax = plt.subplots(figsize=(8,8))
cmap = colors.ListedColormap(['lightgrey', 'steelblue', 'lightgrey'])
ax.imshow(worldmap, cmap=cmap, origin='lower')
ax.scatter(startNode[0], startNode[1], marker="X", color="red", s=200)
ax.scatter(goalNode[0], goalNode[1], marker="X", color="blue", s=200)
ax.scatter(ox_2, oy_2, marker=".", color="black", s=50)
ax.annotate("  Start", (startNode[0], startNode[1]))
ax.annotate("  Goal  ", (goalNode[0], goalNode[1]))
ax.xaxis.set_ticks(np.arange(0, 60, 1))
plt.setp(ax.get_xticklabels(), rotation=90, ha='center')
ax.yaxis.set_ticks(np.arange(0, 60, 1))
plt.plot(GF_xc, GF_yc, color="red")
for i, value in enumerate(GF_distance_value):
    ax.annotate(value, (GF_neighbors_cords[i][0], GF_neighbors_cords[i][1]), color="red")
plt.plot(GF_xc, GF_yc, color = "red")
plt.legend(['Grassfire'], loc=1)
plt.grid(True)
plt.axis("equal")     
plt.show()

# shows the MGF path
fig, ax = plt.subplots(figsize=(8,8))
cmap = colors.ListedColormap(['lightgrey', 'steelblue', 'lightgrey'])
ax.imshow(worldmap, cmap=cmap, origin='lower')
ax.scatter(startNode[0], startNode[1], marker="X", color="red", s=200)
ax.scatter(goalNode[0], goalNode[1], marker="X", color="blue", s=200)
ax.scatter(ox_2, oy_2, marker=".", color="black", s=50)
ax.annotate("  Start", (startNode[0], startNode[1]))
ax.annotate("  Goal  ", (goalNode[0], goalNode[1]))
ax.xaxis.set_ticks(np.arange(0, 60, 1))
plt.setp(ax.get_xticklabels(), rotation=90, ha='center')
ax.yaxis.set_ticks(np.arange(0, 60, 1))
for i, value in enumerate(MGF_distance_value):
    ax.annotate(value, (MGF_neighbors_cords[i][0], MGF_neighbors_cords[i][1]), color="green")
plt.plot(MGF_xc, MGF_yc, color = "green")
plt.legend(['MGrassfire'], loc=1)
plt.grid(True)
plt.axis("equal")     
plt.show() 

# shows both the GF and MGF paths
fig, ax = plt.subplots(figsize=(8,8))
cmap = colors.ListedColormap(['lightgrey', 'steelblue', 'lightgrey'])
ax.imshow(worldmap, cmap=cmap, origin='lower')
ax.scatter(startNode[0], startNode[1], marker="X", color="red", s=200)
ax.scatter(goalNode[0], goalNode[1], marker="X", color="blue", s=200)
ax.scatter(ox_2, oy_2, marker=".", color="black", s=50)
ax.annotate("  Start", (startNode[0], startNode[1]))
ax.annotate("  Goal  ", (goalNode[0], goalNode[1]))
ax.xaxis.set_ticks(np.arange(0, 60, 1))
plt.setp(ax.get_xticklabels(), rotation=90, ha='center')
ax.yaxis.set_ticks(np.arange(0, 60, 1))
for i, value in enumerate(GF_distance_value):
    ax.annotate(("    {value}").format(value=value), (GF_neighbors_cords[i][0], GF_neighbors_cords[i][1]), color="red")
plt.plot(GF_xc, GF_yc, color = "red")
for i, value in enumerate(MGF_distance_value):
    ax.annotate(value, (MGF_neighbors_cords[i][0], MGF_neighbors_cords[i][1]), color="green")
plt.plot(MGF_xc, MGF_yc, color = "green")
plt.legend(['Grassfire', 'MGrassfire'], loc=1)
plt.grid(True)
plt.axis("equal")     
plt.show()

fig, ax = plt.subplots(figsize=(10,10))
cmap = colors.ListedColormap(['lightgrey', 'steelblue', 'lightgrey'])
ax.imshow(worldmap, cmap=cmap, origin='lower')
ax.scatter(startNode[0], startNode[1], marker="X", color="red", s=200)
ax.scatter(goalNode[0], goalNode[1], marker="X", color="blue", s=200)
ax.scatter(ox_2, oy_2, marker=".", color="black", s=50)
ax.scatter(sample_x, sample_y, marker = "x", color ="green", s=50)
ax.annotate("  Start", (startNode[0], startNode[1]))
ax.annotate("  Goal  ", (goalNode[0], goalNode[1]))
ax.xaxis.set_ticks(np.arange(0, 60, 1))
plt.setp(ax.get_xticklabels(), rotation=90, ha='center')
ax.yaxis.set_ticks(np.arange(0, 60, 1))
for i, _ in enumerate(rprm_road_map_matrix):
    for ii in range(len(rprm_road_map_matrix[i])):
        if rprm_road_map_matrix[i][ii] != None:
            plt.plot([sample_x[i], sample_x[ii]], [sample_y[i], sample_y[ii]], "-m")
plt.plot(GF_xc, GF_yc, color = "red")
plt.plot(MGF_xc, MGF_yc, color = "green")
PRM_MGF_path = generate_prm_mgf_path(sample_space_size, GF_xc, rprm_road_map_matrix, sample_x, sample_y)
rprm_x = []
rprm_y = []
for i in range(len(PRM_MGF_path) - 1):
    rprm_x.append(PRM_MGF_path[i][0])
    rprm_y.append(PRM_MGF_path[i][1])
rprm_x.append(goalNode[0])
rprm_y.append(goalNode[1])
plt.plot(rprm_x, rprm_y, color = "black")
plt.legend(['GF', 'MGF', 'r-PRM'], loc=1)
ax = plt.gca()
leg = ax.get_legend()
leg.legendHandles[0].set_color('red')
leg.legendHandles[1].set_color('green')
leg.legendHandles[2].set_color('black')
plt.axis("equal")
plt.grid(True)
plt.show()

fig, ax = plt.subplots(figsize=(8,8))
cmap = colors.ListedColormap(['lightgrey', 'steelblue', 'lightgrey'])
ax.imshow(worldmap, cmap=cmap, origin='lower')
ax.scatter(startNode[0], startNode[1], marker="X", color="red", s=200)
ax.scatter(goalNode[0], goalNode[1], marker="X", color="blue", s=200)
ax.scatter(ox_2, oy_2, marker=".", color="black", s=50)
ax.annotate("  Start", (startNode[0], startNode[1]))
ax.annotate("  Goal  ", (goalNode[0], goalNode[1]))
ax.xaxis.set_ticks(np.arange(0, 60, 1))
plt.setp(ax.get_xticklabels(), rotation=90, ha='center')
ax.yaxis.set_ticks(np.arange(0, 60, 1))
plt.plot(GF_xc, GF_yc, color = "red")
plt.plot(MGF_xc, MGF_yc, color = "green")
plt.plot(rprm_x, rprm_y, color = "black")
plt.legend(['GF', 'MGF', 'r-PRM'], loc=1)
ax = plt.gca()
leg = ax.get_legend()
leg.legendHandles[0].set_color('red')
leg.legendHandles[1].set_color('green')
leg.legendHandles[2].set_color('black')
plt.axis("equal")
plt.grid(True)
plt.show()


########## Local path planner, Start ##########
def nearest_grid_point(x ,y):
    if math.ceil(x) - x <= x - math.floor(x):
        x_1 = math.ceil(x)
    else:
        x_1 = math.floor(x)
    if math.ceil(y) - y <= y - math.floor(y):
        y_1 = math.ceil(y)
    else:
        y_1 = math.floor(y)
    return x_1, y_1

def distanceBetweenPoints(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def go_left_one_step(uuv_x, uuv_y, pux, puy, ux, uy, ox, oy, uuv_path):
    print("going to left of obstacle")
    # determine target point (t1x, t1y): uses method described at
    # https://math.stackexchange.com/questions/175896/finding-a-point-along-a-line-a-certain-distance-away-from-another-point
    print("uuv_x = ", uuv_x, "uuv_y = ", uuv_y)
    print("ux = ", ux, "uy = ", uy)
    print("ox = ", ox, "oy = ", oy)
    slopeOfUuvObLink = (oy - uy) / (ox - ux)
    slopeOfPerpendicular = -1 / slopeOfUuvObLink
    offset = offsetFactor * min_fixed_distance
    print("offset = ", offset)
    # intersection points of perpendicular with circle around ob
    intersection_x_1 = ox + offset / np.sqrt(1 + slopeOfPerpendicular ** 2)
    intersection_y_1 = slopeOfPerpendicular * (intersection_x_1 - ox) + oy
    intersection_x_2 = ox - offset / np.sqrt(1 + slopeOfPerpendicular ** 2)
    intersection_y_2 = slopeOfPerpendicular * (intersection_x_2 - ox) + oy
    print("intersection_x_1 = ", intersection_x_1, "intersection_y_1 = ", intersection_y_1)
    print("intersection_x_2 = ", intersection_x_2, "intersection_y_2 = ", intersection_y_2)
    # choose intersection point to left of line from uuv to ob
    if orient2d((ux, uy), (ox, oy), (intersection_x_1, intersection_y_1)) > 0:
        target_x = intersection_x_1
        target_y = intersection_y_1
    else:
        target_x = intersection_x_2
        target_y = intersection_y_2
    print("target_x = ", target_x, "target_y = ", target_y)
    new_target_x, new_target_y = nearest_grid_point(target_x, target_y)
    print("new_target_x = ", new_target_x, "new_target_y = ", new_target_y)
    connecting_line_segment = get_line_segment_waypoints(ux, uy, new_target_x, new_target_y)[:-1]
    print("connecting_line_segment = ", connecting_line_segment)
    sample_space_size = round(original_sample_space_size * length_prm_mgf_path()/length_original_prm_mgf_path)
    print("new sample_space_size = ", sample_space_size)
    new_global_path = create_new_global_path((new_target_x, new_target_y), goalNode)
    print("new_global_path = ", new_global_path)
    uuv_index = uuv_path.index((uuv_x, uuv_y))
    del uuv_path[uuv_index:len(uuv_path)]
    uuv_path = uuv_path + connecting_line_segment + new_global_path
    print("uuv_path = ", uuv_path)
    return uuv_path, target_x, target_y

def go_left_two_steps(uuv_x, uuv_y, pux, puy, ux, uy, ox, oy, uuv_path):
    print("going to left of obstacle")
    # determine target point (t1x, t1y): uses method described at
    # https://math.stackexchange.com/questions/175896/finding-a-point-along-a-line-a-certain-distance-away-from-another-point
    print("uuv_x = ", uuv_x, "uuv_y = ", uuv_y)
    print("ux = ", ux, "uy = ", uy)
    print("ox = ", ox, "oy = ", oy)
    slopeOfUuvObLink = (oy - uy) / (ox - ux)
    slopeOfPerpendicular = -1 / slopeOfUuvObLink
    offset = offsetFactor * min_fixed_distance
    # intersection points of perpendicular with circle around ob
    intersection_x_1 = ox + offset / np.sqrt(1 + slopeOfPerpendicular ** 2)
    intersection_y_1 = slopeOfPerpendicular * (intersection_x_1 - ox) + oy
    intersection_x_2 = ox - offset / np.sqrt(1 + slopeOfPerpendicular ** 2)
    intersection_y_2 = slopeOfPerpendicular * (intersection_x_2 - ox) + oy
    # choose intersection point to left of line from uuv to ob
    if orient2d((ux, uy), (ox, oy), (intersection_x_1, intersection_y_1)) > 0:
        target_1_x = intersection_x_1
        target_1_y = intersection_y_1
    else:
        target_1_x = intersection_x_2
        target_1_y = intersection_y_2
    print("target_1_x = ", target_1_x, "target_1_y = ", target_1_y)
    line_segment_1 = get_line_segment_waypoints(ux, uy, target_1_x, target_1_y)[:-1]
    print("line_segment_1 = ", line_segment_1)
    # intersection points of uuv-ob line with circle around ob
    intersection_x_3 = ox + offset / np.sqrt(1 + slopeOfUuvObLink ** 2)
    intersection_y_3 = slopeOfUuvObLink * (intersection_x_3 - ox) + oy
    intersection_x_4 = ox - offset / np.sqrt(1 + slopeOfUuvObLink ** 2)
    intersection_y_4 = slopeOfUuvObLink * (intersection_x_4 - ox) + oy
    # choose intersection point to left of line from target_1 to ob,
    # i.e., which is opposite from uuv with respect to ob
    if orient2d((target_1_x, target_1_y), (ox, oy), (intersection_x_3, intersection_y_3)) > 0:
        target_2_x = intersection_x_3
        target_2_y = intersection_y_3
    else:
        target_2_x = intersection_x_4
        target_2_y = intersection_y_4
    print("target_2_x = ", target_2_x, "target_2_y = ", target_2_y)
    new_target_2_x, new_target_2_y = nearest_grid_point(target_2_x, target_2_y)
    print("new_target_2_x = ", new_target_2_x, "new_target_2_y = ", new_target_2_y)
    line_segment_2 = get_line_segment_waypoints(target_1_x, target_1_y, new_target_2_x, new_target_2_y)[:-1]
    print("line_segment_2 = ", line_segment_2)
    sample_space_size = round(original_sample_space_size * length_prm_mgf_path()/length_original_prm_mgf_path)
    print("new sample_space_size = ", sample_space_size)
    new_global_path = create_new_global_path((new_target_2_x, new_target_2_y), goalNode)
    uuv_index = uuv_path.index((uuv_x, uuv_y))
    del uuv_path[uuv_index:len(uuv_path)]
    uuv_path = uuv_path + line_segment_1 + line_segment_2 + new_global_path
    print("uuv_path = ", uuv_path)
    return uuv_path, new_target_2_x, new_target_2_y

def go_right_one_step(uuv_x, uuv_y, pux, puy, ux, uy, ox, oy, uuv_path):
    print("going to right of obstacle")
    # determine target point (t1x, t1y): uses method described at
    # https://math.stackexchange.com/questions/175896/finding-a-point-along-a-line-a-certain-distance-away-from-another-point
    print("uuv_x = ", uuv_x, "uuv_y = ", uuv_y)
    print("ux = ", ux, "uy = ", uy)
    print("ox = ", ox, "oy = ", oy)
    slopeOfUuvObLink = (oy - uy) / (ox - ux)
    slopeOfPerpendicular = -1 / slopeOfUuvObLink
    offset = offsetFactor * min_fixed_distance
    # intersection points of perpendicular with circle around ob
    intersection_x_1 = ox + offset / np.sqrt(1 + slopeOfPerpendicular ** 2)
    intersection_y_1 = slopeOfPerpendicular * (intersection_x_1 - ox) + oy
    intersection_x_2 = ox - offset / np.sqrt(1 + slopeOfPerpendicular ** 2)
    intersection_y_2 = slopeOfPerpendicular * (intersection_x_2 - ox) + oy
    # choose intersection point to right of line from uuv to ob
    if orient2d((ux, uy), (ox, oy), (intersection_x_1, intersection_y_1)) < 0:
        target_x = intersection_x_1
        target_y = intersection_y_1
    else:
        target_x = intersection_x_2
        target_y = intersection_y_2
    print("target_x = ", target_x, "target_y = ", target_y)
    new_target_x, new_target_y = nearest_grid_point(target_x, target_y)
    connecting_line_segment = get_line_segment_waypoints(ux, uy, new_target_x, new_target_y)[:-1]
    sample_space_size = round(original_sample_space_size * length_prm_mgf_path()/length_original_prm_mgf_path)
    print("new sample_space_size = ", sample_space_size)
    new_global_path = create_new_global_path((new_target_x, new_target_y), goalNode)
    uuv_index = uuv_path.index((uuv_x, uuv_y))
    del uuv_path[uuv_index:len(uuv_path)]
    uuv_path = uuv_path + connecting_line_segment + new_global_path
    print("uuv_path = ", uuv_path)
    return uuv_path, new_target_x, new_target_y

def go_right_two_steps(uuv_x, uuv_y, pux, puy, ux, uy, ox, oy, uuv_path):
    print("going to right of obstacle")
    # determine target point (t1x, t1y): uses method described at
    # https://math.stackexchange.com/questions/175896/finding-a-point-along-a-line-a-certain-distance-away-from-another-point
    print("uuv_x = ", uuv_x, "uuv_y = ", uuv_y)
    print("ux = ", ux, "uy = ", uy)
    print("ox = ", ox, "oy = ", oy)
    slopeOfUuvObLink = (oy - uy) / (ox - ux)
    slopeOfPerpendicular = -1 / slopeOfUuvObLink
    offset = offsetFactor * min_fixed_distance
    # intersection points of perpendicular with circle around ob
    intersection_x_1 = ox + offset / np.sqrt(1 + slopeOfPerpendicular ** 2)
    intersection_y_1 = slopeOfPerpendicular * (intersection_x_1 - ox) + oy
    intersection_x_2 = ox - offset / np.sqrt(1 + slopeOfPerpendicular ** 2)
    intersection_y_2 = slopeOfPerpendicular * (intersection_x_2 - ox) + oy
    # choose intersection point to right of line from uuv to ob
    if orient2d((ux, uy), (ox, oy), (intersection_x_1, intersection_y_1)) < 0:
        target_1_x = intersection_x_1
        target_1_y = intersection_y_1
    else:
        target_1_x = intersection_x_2
        target_1_y = intersection_y_2
    print("target_1_x = ", target_1_x, "target_1_y = ", target_1_y)
    line_segment_1 = get_line_segment_waypoints(ux, uy, target_1_x, target_1_y)[:-1]
    print("line_segment_1 = ", line_segment_1)
    # intersection points of uuv-ob line with circle around ob
    intersection_x_3 = ox + offset / np.sqrt(1 + slopeOfUuvObLink ** 2)
    intersection_y_3 = slopeOfUuvObLink * (intersection_x_3 - ox) + oy
    intersection_x_4 = ox - offset / np.sqrt(1 + slopeOfUuvObLink ** 2)
    intersection_y_4 = slopeOfUuvObLink * (intersection_x_4 - ox) + oy
    # choose intersection point to right of line from target_1 to ob,
    # i.e., which is opposite from uuv with respect to ob
    if orient2d((target_1_x, target_1_y), (ox, oy), (intersection_x_3, intersection_y_3)) < 0:
        target_2_x = intersection_x_3
        target_2_y = intersection_y_3
    else:
        target_2_x = intersection_x_4
        target_2_y = intersection_y_4
    print("target_2_x = ", target_2_x, "target_2_y = ", target_2_y)
    new_target_2_x, new_target_2_y = nearest_grid_point(target_2_x, target_2_y)
    print("new_target_2_x = ", new_target_2_x, "new_target_2_y = ", new_target_2_y)
    line_segment_2 = get_line_segment_waypoints(target_1_x, target_1_y, new_target_2_x, new_target_2_y)[:-1]
    print("line_segment_2 = ", line_segment_2)
    sample_space_size = round(original_sample_space_size * length_prm_mgf_path()/length_original_prm_mgf_path)
    print("new sample_space_size = ", sample_space_size)
    new_global_path = create_new_global_path((new_target_2_x, new_target_2_y), goalNode)
    uuv_index = uuv_path.index((uuv_x, uuv_y))
    del uuv_path[uuv_index:len(uuv_path)]
    uuv_path = uuv_path + line_segment_1 + line_segment_2 + new_global_path
    return uuv_path, new_target_2_x, new_target_2_y

def local_path_planner(uuv_x, uuv_y, uuv_x_previous, uuv_y_previous,
                       ob_x, ob_y, ob_x_previous, ob_y_previous, uuv_path):
    print("in local path planner")

    # Rationale: possible conditions of moving object relative to uuv and actions to take in each situation.
    #
    # uuv path and ob path are parallel
    #   ob left of or on uuv path
    #     go right one step
    #   ob right of uuv path
    #     go left one step
    #
    # uuv path and ob path intersect behind uuv (intersection point is left of left perpendicular to uuv path at uuv) or
    #     at uuv (intersection point is on left perpendicular to uuv path at uuv)
    #   ob (left of or on uuv path) and ob moving toward uuv and uuv moving toward ob path
    #     go right one step
    #   ob right of uuv path and ob moving toward uuv and uuv moving toward ob path
    #     go left one step
    #
    # uuv path and ob path intersect ahead of uuv (intersection point is right of left perpendicular to uuv path at uuv)
    #   ob left of or on uuv path
    #     ob moving toward uuv path
    #       go left two steps (behind ob)
    #     ob moving away from uuv path
    #       go right one step
    #   ob right of uuv path
    #     ob moving toward uuv path
    #       go right two steps (behind ob)
    #     ob moving away from uuv path
    #       go left one step

    ux = uuv_x / 20
    uy = uuv_y / 20
    pux = uuv_x_previous / 20
    puy = uuv_y_previous / 20
    if pux != ux:
        slopeUuvPath = (puy - uy) / (pux - ux)
    else:
        slopeUuvPath = math.inf
    print("ux = ", ux, "uy = ", uy)
    print("pux = ", pux, "puy = ", puy)
    print("slopeUuvPath = ", slopeUuvPath)
    ox = ob_x
    oy = ob_y
    pox = ob_x_previous
    poy = ob_y_previous
    if pox != ox:
        slopeObPath = (poy - oy) / (pox - ox)
    else:
        slopeObPath = math.inf
    print("ox = ", ox, "oy = ", oy)
    print("pox = ", pox, "poy = ", poy)
    print("slopeObPath = ", slopeObPath)

    obVersusUuv = orient2d((pux, puy), (ux, uy), (ox, oy))
    obIsLeftOfUuvPath = (obVersusUuv > 0)
    obIsRightOfUuvPath = (obVersusUuv < 0)
    obIsOnUuvPath = (obVersusUuv == 0)

    if slopeUuvPath == slopeObPath:
        print("uuv path and ob path are parallel")
        if obIsLeftOfUuvPath or obIsOnUuvPath:
            print("ob is left of or on uuv path")
            print("go right one step")
            uuv_path, tx, ty = go_right_one_step(uuv_x, uuv_y, pux, puy, ux, uy, ox, oy, uuv_path)
            return uuv_path, tx, ty
        if obIsRightOfUuvPath:
            print("ob is right of uuv path")
            print("go left one step")
            uuv_path, tx, ty = go_left_one_step(uuv_x, uuv_y, pux, puy, ux, uy, ox, oy, uuv_path)
            return uuv_path, tx, ty
    else:
        print("uuv path and ob path not parallel")

    # Find intersection (ix,iy) of uuv path and ob path, given that the paths are not parallel.
    #
    # For any two points (x1,y1) and (x2,y2), if x1!=x2, the equation of the line through the points is
    # (y-y1)/(x-x1)=m, where m=(y2-y1)/(x2-x1) is the slope.  Solving gives y=x*m+(y1-x1*m).  If x1=x2, , meaning
    # that the line is vertical, the equation is just x=x1 (or x=x2).
    #
    # Thus, if pux!=ux the equation of the uuv path is
    #   y=x*slopeUuvPath+(uy-ux*slopeUuvPath)    (1)
    # and x=ux otherwise.
    #
    # Similarly, if pox!=ox, the equation of the ob path is
    #   y=x*slopeObPath+(oy-ox*slopeObPath)      (2)
    #   and x=ox otherwise.
    #
    # To find the intersection point (ix,iy) of two lines given as y=m1*x+b1 and y=m2*x+b2, set the two right sides equal
    # and solve for x. This is ix. Then substitute this ix into either of the line equations to determine iy.
    # Doing this for these line equations yields ix=(b2-b1)/(m1-m2) and iy=m1*ix+b1. Note that m1!=m2 as long as the
    # lines are not parallel (which we have assumed).  If the first of the two lines is vertical, with equation x=c, set
    # ix=c and iy=m2*xi+b2. Similarly, if the second of the two lines is vertical, with equation x=c, set ix=c and
    # iy=m1*ix+b1.
    #
    # Applying this to the uuv and ob paths gives the following. Again, it is agreed that these are not parallel.
    # Case (a): ux!=pux and ox!=pox.  Setting the right sides of (1) and (2) equal and solving for x gives
    #   ix=[(oy-ox*slopeObPath)-(uy-ux*slopeUuvPath)]/(slopeUuvPath-slopeObPath)
    # Substituting ix into (1) gives
    #   iy=ix*slopeUuvPath+(uy-ux*slopeUuvPath)
    # Case (b): ux=pux and ox!=pox (slope UuvPath = math.inf). Then set
    #   ix=ux
    #   iy=ix*slopeObPath+(oy-ox*slopeObPath)
    # Case (c) ux!=pux and ox=pox (slope ObPath = math.inf). Then set
    #   ix=ox
    #   iy=ix*slopeUuvPath+(oy-ox*slopeUuvPath)

    if ux != pux and ox != pox:
        print("case a")
        ix = [(oy - ox * slopeObPath) - (uy - ux * slopeUuvPath)] / (slopeUuvPath - slopeObPath)
        iy = ix * slopeUuvPath + (puy - pux * slopeUuvPath)
    elif ux == pux and ox != pox:
        print("case b")
        ix = ux
        iy = ix * slopeObPath + (oy - ox * slopeObPath)
    elif ux != pux and ox == pox:
        print("case c")
        ix = ox
        iy = ix * slopeUuvPath + (uy - ux * slopeUuvPath)
    print("ix = ", ix, "iy = ", iy)

    # Construct a reference point (rx, ry) that is to the left of the uuv path and on a line perpendicular to the path
    # through the point (ux, uy).  This is done as follows.
    # Case (a) pux=ux. Then the uuv path is vertical, and the perpendicular is horizontal. Note that in this case puy!=uy,
    #   else the two points are identical (which is not possible in this simulation).
    #   Subcase (i) puy<uy, so the path is pointing up. Let ry=uy and rx=ux-0.05.
    #   Subcase (ii) puy>uy, so the path is pointing down. Let ry=uy and rx=ux+0.05.
    # Case (b) pux!=ux.
    #   Subcase (i) slopeUuvPath=0. Then the uuv path is horizontal and the perpendicular is vertical. Let rx=ux and
    #     ry=uy+0.05.
    #   Subcase (ii) slopeUuvPath!=0. Then the slope of the perpendicular is mp=-1/slopeUuvPath.
    #     For (x,y) an arbitrary point on the perpendicular through (ux,uy), we have mp = (y-uy)/(x-ux), and solving this
    #     gives the equation of the perpendicular as y=mp*(x-ux)+uy.  If mp<0, let rx=ux-0.05.  If mp>0, let rx=ux+0.05.
    #     In either case, let ry=mp*(rx-ux)+uy.  If mp<0, this reduces to ry=-0.05*mp+uy.  If mp>0, this reduces to
    #     ry=0.05*mp+uy.

    if pux == ux:
        if puy < uy:
            rx = ux - 0.05
            ry = uy
        elif puy > uy:
            rx = ux + 0.05
            ry = uy
    else:
        if slopeUuvPath == 0:
            rx = ux
            ry = uy + 0.05
        else:
            mp = -1 / slopeUuvPath
            if mp < 0:
                rx = ux - 0.05
                ry = -0.05 * mp + uy
            else:
                rx = ux + 0.05
                ry = 0.05 * mp + uy
    print("rx = ", rx, " ry = ", ry)
    intersectionPointVersusLeftPerpendicular = orient2d((ux, uy), (rx, ry), (ix, iy))
    print("intersectionPointVersusLeftPerpendicular = ", intersectionPointVersusLeftPerpendicular)
    intersectionPointIsBehindUuv = (intersectionPointVersusLeftPerpendicular > 0)
    print("intersectionPointIsBehindUuv = ", intersectionPointIsBehindUuv)
    intersectionPointIsAheadOfUuv = (intersectionPointVersusLeftPerpendicular < 0)
    print("intersectionPointIsAheadOfUuv = ", intersectionPointIsAheadOfUuv)
    intersectionPointIsAtUuv = (intersectionPointVersusLeftPerpendicular == 0)
    print("intersectionPointIsAtUuv = ", intersectionPointIsAtUuv)

    # In the case that the uuv path and ob path intersect ahead of the uuv or intersect behind uuv, determine whether the ob is moving toward or
    # away from the uuv path. The ob is moving toward the uuv path if (ox,oy) is closer to the uuv path than (pox,poy) and
    # is moving away from the uuv path if (pox,poy) is closer to the uuv path than (ox,oy).  Note that the two points can't
    # be the same distance from the uuv path since the two paths are here known to intersect.
    #
    # From https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line, If the line passes through two points
    # P1 = (x1, y1) and P2 = (x2, y2) then the distance of (x0, y0) from the line is:
    #   distance(P1,P2,(x0,y0))=abs[(x2-x1)(y1-y0)-(x1-x0)(y2-y1)]/distanceBetween(P1,P2)
    # Let (pux,puy) be P1, (ux,uy) be P2.  Then taking (pox,poy) for (x0,y0) gives
    #   distancePobFromUuvPath=abs[(ux-pux)(uy-poy)-(pux-pox)(uy-puy)]/distanceBetweenPoints(pux,puy,ux,uy)
    # and taking (ox,oy) for (x0,y0) gives
    #   distanceObFromUuvPath=abs[(ux-pux)(uy-oy)-(pux-ox)(uy-puy)]/distanceBetweenPoints(pux,puy,ux,uy)

    pobVersusUuv = orient2d((pux, puy), (ux, uy), (pox, poy))
    pobIsLeftOfUuvPath = (pobVersusUuv > 0)
    pobIsRightOfUuvPath = (pobVersusUuv < 0)
  #  pobIsOnUuvPath = (pobVersusUuv == 0)

    if pobIsLeftOfUuvPath and obIsLeftOfUuvPath:
        distancePobFromUuvPath = abs((ux - pux) * (puy - poy) - (pux - pox) * (uy - puy)) / distanceBetweenPoints(pux, puy, ux, uy)
        distanceObFromUuvPath = abs((ux - pux) * (puy - oy) - (pux - ox) * (uy - puy)) / distanceBetweenPoints(pux, puy, ux, uy)
        print("distancePobFromUuvPath = ", distancePobFromUuvPath)
        print("distanceObFromUuvPath = ", distanceObFromUuvPath)
        obMovingTowardUuvPath = (distancePobFromUuvPath > distanceObFromUuvPath)
        obMovingAwayFromUuvPath = (distancePobFromUuvPath < distanceObFromUuvPath)
        print("obMovingTowardUuvPath = ", obMovingTowardUuvPath)
        print("obMovingAwayFromUuvPath = ", obMovingAwayFromUuvPath)
    elif pobIsRightOfUuvPath and obIsRightOfUuvPath:
        distancePobFromUuvPath = abs((ux - pux) * (puy - poy) - (pux - pox) * (uy - puy)) / distanceBetweenPoints(pux, puy, ux, uy)
        distanceObFromUuvPath = abs((ux - pux) * (puy - oy) - (pux - ox) * (uy - puy)) / distanceBetweenPoints(pux, puy, ux, uy)
        print("distancePobFromUuvPath = ", distancePobFromUuvPath)
        print("distanceObFromUuvPath = ", distanceObFromUuvPath)
        obMovingTowardUuvPath = (distancePobFromUuvPath > distanceObFromUuvPath)
        obMovingAwayFromUuvPath = (distancePobFromUuvPath < distanceObFromUuvPath)
        print("obMovingTowardUuvPath = ", obMovingTowardUuvPath)
        print("obMovingAwayFromUuvPath = ", obMovingAwayFromUuvPath)
    else:
        obMovingTowardUuvPath = False
        obMovingAwayFromUuvPath = True
        print("obMovingTowardUuvPath = ", obMovingTowardUuvPath)
        print("obMovingAwayFromUuvPath = ", obMovingAwayFromUuvPath)

    distancePuuvFromObPath = abs((ox - pox) * (poy - puy) - (pox - pux) * (oy - poy)) / distanceBetweenPoints(pox, poy, ox, oy)
    distanceUuvFromObPath = abs((ox - pox) * (poy - uy) - (pox - ux) * (oy - poy)) / distanceBetweenPoints(pox, poy, ox, oy)
    print("distancePuuvFromObPath = ", distancePuuvFromObPath)
    print("distanceUuvFromObPath = ", distanceUuvFromObPath)
    uuvMovingTowardObPath = (distancePuuvFromObPath > distanceUuvFromObPath)
  #  uuvMovingawayFromObPath = (distancePuuvFromObPath < distanceUuvFromObPath)

    if intersectionPointIsBehindUuv or intersectionPointIsAtUuv:
        print("intersectionPointIsBehindUuv or intersectionPointIsAtUuv")
        if (obIsLeftOfUuvPath or obIsOnUuvPath) and obMovingTowardUuvPath and uuvMovingTowardObPath:
            print("(obIsLeftOfUuvPath or obIsOnUuvPath) and obMovingTowardUuvPath and uuvMovingTowardObPath")
            print("go right one step")
            uuv_path, tx, ty = go_right_one_step(uuv_x, uuv_y, pux, puy, ux, uy, ox, oy, uuv_path)
            print("tx = ", tx, "ty = ", ty)
            return uuv_path, tx, ty
        if obIsRightOfUuvPath and obMovingTowardUuvPath and uuvMovingTowardObPath:
            print("obIsRightOfUuvPath and obMovingTowardUuvPath and uuvMovingTowardObPath")
            print("go left one step")
            uuv_path, tx, ty = go_left_one_step(uuv_x, uuv_y, pux, puy, ux, uy, ox, oy, uuv_path)
            print("tx = ", tx, "ty = ", ty)
            return uuv_path, tx, ty
        return uuv_path, ux, uy

    if intersectionPointIsAheadOfUuv:
        print("intersectionPointIsAheadOfUuv")
        if obIsLeftOfUuvPath or obIsOnUuvPath:
            print("obIsLeftOfUuvPath or obIsOnUuvPath")
            if obMovingTowardUuvPath:
                print("obMovingTowardUuvPath")
                print("go left two steps")
                uuv_path, tx, ty = go_left_two_steps(uuv_x, uuv_y, pux, puy, ux, uy, ox, oy, uuv_path)
                print("tx = ", tx, "ty = ", ty)
                return uuv_path, tx, ty
            elif obMovingAwayFromUuvPath:
                print("obMovingAwayFromUuvPath")
                print("go right one step")
                uuv_path, tx, ty = go_right_one_step(uuv_x, uuv_y, pux, puy, ux, uy, ox, oy, uuv_path)
                return uuv_path, tx, ty
        if obIsRightOfUuvPath:
            print("obIsRightOfUuvPath")
            if obMovingTowardUuvPath:
                print("obMovingTowardUuvPath")
                print("go right two steps")
                uuv_path, tx, ty = go_right_two_steps(uuv_x, uuv_y, pux, puy, ux, uy, ox, oy, uuv_path)
                return uuv_path, tx, ty
            elif obMovingAwayFromUuvPath:
                print("obMovingAwayFromUuvPath")
                print("go left one step")
                uuv_path, tx, ty = go_left_one_step(uuv_x, uuv_y, pux, puy, ux, uy, ox, oy, uuv_path)
                return uuv_path, tx, ty
########## Local path planner, End ##########

########## Mobile Objects, Start ##########
mob_1_path, mob_2_path, mob_3_path, mob_4_path, mob_5_path, uuv_path = 0, 0, 0, 0, 0, 0
steer = 0.6  # [rad] maximum steering angle

timestamp = [0, 10, 20, 30]
intervals = np.linspace(0, 30, 150)

#x1 = [10, 10, 10, 10]
#y1 = [56.00, 52.0, 48.0, 44.0]

x1 = [10, 10, 10, 10]
y1 = [54.00, 53.0, 53.5, 52.0]

iplt_x1 = np.interp(intervals, timestamp, x1)
iplt_y1 = np.interp(intervals, timestamp, y1)
iplt_x1_b = list(iplt_x1)
iplt_y1_b = list(iplt_y1)

path_mob_1 = list(zip(iplt_x1, iplt_y1))
mob_1_path = 0
for i in (range(0, len(path_mob_1) - 1)):
    mob_1_path = mob_1_path + math.sqrt(
        (path_mob_1[i + 1][0] - path_mob_1[i][0]) ** 2 + (path_mob_1[i + 1][1] - path_mob_1[i][1]) ** 2)

mob_1_speed = mob_1_path / 150  # speed = 1 m/s
angle1 = mob_1_path * tan(steer) / 0.15  # r=0.15
yaw1 = (angle1 + pi) % (2 * pi) - pi

x2 = [20, 23, 25, 28]
y2 = [40, 43, 45, 48]

iplt_x2 = np.interp(intervals, timestamp, x2)
iplt_y2 = np.interp(intervals, timestamp, y2)
iplt_x2_b = list(iplt_x2)
iplt_y2_b = list(iplt_y2)

path_mob_2 = list(zip(iplt_x2, iplt_y2))
mob_2_path = 0
for i in (range(0, len(path_mob_2) - 1)):
    mob_2_path = mob_2_path + math.sqrt(
        (path_mob_2[i + 1][0] - path_mob_2[i][0]) ** 2 + (path_mob_2[i + 1][1] - path_mob_2[i][1]) ** 2)

mob_2_speed = mob_2_path / 150
angle2 = mob_2_path * tan(steer) / 0.15  # r=0.15
yaw2 = (angle2 + pi) % (2 * pi) - pi

x3 = [35, 34, 33, 32]
y3 = [38, 37, 36, 35]

iplt_x3 = np.interp(intervals, timestamp, x3)
iplt_y3 = np.interp(intervals, timestamp, y3)
iplt_x3_b = list(iplt_x3)
iplt_y3_b = list(iplt_y3)

path_mob_3 = list(zip(iplt_x3, iplt_y3))
mob_3_path = 0
for i in (range(0, len(path_mob_3) - 1)):
    mob_3_path = mob_3_path + math.sqrt(
        (path_mob_3[i + 1][0] - path_mob_3[i][0]) ** 2 + (path_mob_3[i + 1][1] - path_mob_3[i][1]) ** 2)

mob_3_speed = mob_3_path / 150
angle3 = mob_3_path * tan(steer) / 0.15  # r=0.15
yaw3 = (angle3 + pi) % (2 * pi) - pi

x4 = [34.8, 36, 37.2, 36.0]
y4 = [18.0, 19.2, 20, 19.2]

iplt_x4 = np.interp(intervals, timestamp, x4)
iplt_y4 = np.interp(intervals, timestamp, y4)
iplt_x4_b = list(iplt_x4)
iplt_y4_b = list(iplt_y4)

path_mob_4 = list(zip(iplt_x4, iplt_y4))
mob_4_path = 0
for i in (range(0, len(path_mob_4) - 1)):
    mob_4_path = mob_4_path + math.sqrt(
        (path_mob_4[i + 1][0] - path_mob_4[i][0]) ** 2 + (path_mob_4[i + 1][1] - path_mob_4[i][1]) ** 2)

mob_4_speed = mob_4_path / 150
angle4 = mob_4_path * tan(steer) / 0.15  # r=0.15
yaw4 = (angle4 + pi) % (2 * pi) - pi

x5 = [46, 46, 46, 46]
y5 = [12, 15, 17, 20]

iplt_x5 = np.interp(intervals, timestamp, x5)
iplt_y5 = np.interp(intervals, timestamp, y5)
iplt_x5_b = list(iplt_x5)
iplt_y5_b = list(iplt_y5)

path_mob_5 = list(zip(iplt_x5, iplt_y5))
mob_5_path = 0
for i in (range(0, len(path_mob_5) - 1)):
    mob_5_path = mob_5_path + math.sqrt(
        (path_mob_5[i + 1][0] - path_mob_5[i][0]) ** 2 + (path_mob_5[i + 1][1] - path_mob_5[i][1]) ** 2)

mob_5_speed = mob_5_path / 150
angle5 = mob_5_path * tan(steer) / 0.15  # r=0.15
yaw5 = (angle5 + pi) % (2 * pi) - pi

path_xc = []
path_yc = []
prm_astar_timestamp = []
for i in (range(0,len(global_path))):
    x = global_path[i][0]
    y = global_path[i][1]
    prm_astar_timestamp.append(5*i)
    path_xc.append(x)
    path_yc.append(y) 
path_xc.reverse()
path_yc.reverse()
prm_astar_intervals = np.linspace(0, 5*len(global_path), 7*150)
iplt_xc = np.interp(prm_astar_intervals, prm_astar_timestamp, path_yc)
iplt_yc = np.interp(prm_astar_intervals, prm_astar_timestamp, path_xc)

path_uuv = list(zip(iplt_xc, iplt_yc))
remus_path = 0
for i in (range(0, len(path_uuv) - 1)):
    remus_path = remus_path + math.sqrt(
        (path_uuv[i + 1][0] - path_uuv[i][0]) ** 2 + (path_uuv[i + 1][1] - path_uuv[i][1]) ** 2)

uuv_speed = remus_path / (7*150)
angleUUV = remus_path * tan(steer) / 0.15  # r=0.15
yawUUV = (angle5 + pi) % (2 * pi) - pi


ob1_xwaypoints = []
ob1_ywaypoints = []
ob2_xwaypoints = []
ob2_ywaypoints = []
ob3_xwaypoints = []
ob3_ywaypoints = []
ob4_xwaypoints = []
ob4_ywaypoints = []
ob5_xwaypoints = []
ob5_ywaypoints = []
for i in range(7):
    for ox1, oy1, ox2, oy2, ox3, oy3, ox4, oy4, ox5, oy5, in zip(iplt_x1_b, iplt_y1_b, iplt_x2_b, iplt_y2_b, iplt_x3_b,
                                                                 iplt_y3_b, iplt_x4_b, iplt_y4_b, iplt_x5_b, iplt_y5_b):
        ob1_xwaypoints.append(ox1)
        ob1_ywaypoints.append(oy1)
        ob2_xwaypoints.append(ox2)
        ob2_ywaypoints.append(oy2)
        ob3_xwaypoints.append(ox3)
        ob3_ywaypoints.append(oy3)
        ob4_xwaypoints.append(ox4)
        ob4_ywaypoints.append(oy4)
        ob5_xwaypoints.append(ox5)
        ob5_ywaypoints.append(oy5)
    iplt_x1_b = iplt_x1_b[::-1]
    iplt_y1_b = iplt_y1_b[::-1]
    iplt_x2_b = iplt_x2_b[::-1]
    iplt_y2_b = iplt_y2_b[::-1]
    iplt_x3_b = iplt_x3_b[::-1]
    iplt_y3_b = iplt_y3_b[::-1]
    iplt_x4_b = iplt_x4_b[::-1]
    iplt_y4_b = iplt_y4_b[::-1]
    iplt_x5_b = iplt_x5_b[::-1]
    iplt_y5_b = iplt_y5_b[::-1]
########## Mobile objects, End ##########

#############################  D Star Start ###################################
class State:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.state = "."  # new state
        self.t = "new"  # tag for state
        self.h = 0
        self.k = 0

    def cost(self, state):
        if self.state == "#" or state.state == "#":          # obstacle state
            return maxsize

        return math.sqrt(math.pow((self.x - state.x), 2) +
                         math.pow((self.y - state.y), 2))

    def set_state(self, state):
        """
        .: new
        #: obstacle
        e: oparent of current state
        *: closed state
        s: current state
        """
        if state not in ["s", ".", "#", "e", "*"]:
            return
        self.state = state


class Map:
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.map = self.init_map()

    def init_map(self):
        map_list = []
        for i in range(self.row):
            tmp = []
            for j in range(self.col):
                tmp.append(State(i, j))
            map_list.append(tmp)
        return map_list

    def get_neighbors(self, state):
        state_list = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0:
                    continue
                if state.x + i < 0 or state.x + i >= self.row:
                    continue
                if state.y + j < 0 or state.y + j >= self.col:
                    continue
                state_list.append(self.map[state.x + i][state.y + j])
        return state_list

    def set_obstacle(self, point_list):
        for x, y in point_list:
            if x < 0 or x >= self.row or y < 0 or y >= self.col:
                continue
            self.map[x][y].set_state("#")  # obstacle state   
        
class Dstar:
    def __init__(self, maps):
        self.map = maps
        self.open_list = set()

    def process_state(self):
        x = self.min_state()

        if x is None:
            return -1

        k_old = self.get_kmin()   
        self.remove(x)    # state X with the lowest kmin value is removed from the OPEN list 

        # state X is a RAISE state (k_old < x.h) means that on the OPEN list to propagate information 
        # about path cost increases (e.g., due to an increased arc cost), its path cost may not be 
        # optimal. Before state X propagates cost changes to its neighbors, its optimal neighbors are 
        # examined to see if x.h can be reduced.
        if k_old < x.h:
            for y in self.map.get_neighbors(x):
                if y.h <= k_old and x.h > y.h + x.cost(y):
                    x.parent = y
                    x.h = y.h + x.cost(y)
       
        # state X is a LOWER state (k_old == x.h) means that on the OPEN list to propagate information 
        # about path cost reductions (e.g., due to a reduced arc cost or new path to the goal). Each 
        # neighbor Y of X is examined to see if its path cost can be lowered.   
        # Additionally, neighbor states that are NEW receive an initial path cost value, and cost 
        # changes are propagated to each neighbor Y that has a backpointer to X, regardless of whether
        # the new cost is greater than or less than the old. Since these states are descendants of X, 
        # any change to the path cost of X affects their path costs as well.   
        # The backpointer of Y is redirected (if needed) so that the monotonic sequence {Y} is 
        # constructed. All neighbors that receive a new path cost are placed on the OPEN list, so that
        # they will propagate the cost changes to their neighbors.
        elif k_old == x.h:                                             
            for y in self.map.get_neighbors(x):       
                if y.t == "new" or y.parent == x and y.h != x.h + x.cost(y) \
                        or y.parent != x and y.h > x.h + x.cost(y):
                    y.parent = x
                    self.insert(y, x.h + x.cost(y))
        else:
            # cost changes are propagated to NEW states and immediate descendants in the same way as 
            # for LOWER states. 
            for y in self.map.get_neighbors(x):
                if y.t == "new" or y.parent == x and y.h != x.h + x.cost(y):
                    y.parent = x
                    self.insert(y, x.h + x.cost(y))
                else:
                    # If state X is able to lower the path cost of a state that is not an
                    # immediate descendant, X is placed back on the OPEN list for future expansion. 
                    # This action is required to avoid creating a closed loop in the backpointers. 
                    if y.parent != x and y.h > x.h + x.cost(y):
                        self.insert(y, x.h)
                    else:
                        # If the path cost of state X is able to be reduced by a suboptimal neighbor, 
                        # the neighbor is placed back on the OPEN list. Thus, the update is postponed
                        # until the neighbor has an optimal path cost.
                        if y.parent != x and x.h > y.h + x.cost(y) and y.t == "close" and y.h > k_old:
                            self.insert(y, y.h)
        return self.get_kmin()
   
    # min_state returns the state on the OPEN list with minimum path cost value k 
    # (NULL if the OPEN list is empty)
    def min_state(self):
        if not self.open_list:
            return None
        min_state = min(self.open_list, key=lambda x: x.k)
        return min_state

    # get_min returns the minimum path cost value k 
    # (-1 if the OPEN list is empty)
    def get_kmin(self):
        if not self.open_list:
            return -1
        k_min = min([x.k for x in self.open_list])
        return k_min

    # insert adds state to the OPEN list 
    def insert(self, state, h_new):
        if state.t == "new":
            state.k = h_new
        elif state.t == "open":
            state.k = min(state.k, h_new)
        elif state.t == "close":
            state.k = min(state.h, h_new)
        state.h = h_new
        state.t = "open"
        self.open_list.add(state)
   
    # remove deletes state from the OPEN list 
    def remove(self, state):
        if state.t == "open":
            state.t = "close"
        self.open_list.remove(state)

    # Since the path cost for the state will change, the state is placed on the OPEN list and is 
    # expanded via process_state(). Hence, the arc cost function is updated with the changed value.
    def modify(self, state):
        self.modify_cost(state)
        while True:
            k_min = self.process_state()
            if k_min >= state.h:
                break  

    def modify_cost(self, x):
        if x.t == "close":
            self.insert(x, x.parent.h + x.cost(x.parent))

    def run(self, start, end):
        dstar_x = []
        dstar_y = []
        self.open_list.add(end) # Goal node placed initially

        while True:
            self.process_state()
            if start.t == "close":
                break

        start.set_state("s")  # current state
        start.parent.set_state("e")    # oparent of current state
        tmp = start
        while tmp != end:
            tmp.set_state("*") # closed state
            dstar_x.append(tmp.x)
            dstar_y.append(tmp.y)

            global time
            uuv_to_ob1_dist = math.sqrt((tmp.x - ob1_xwaypoints[time]) ** 2 + (tmp.y - ob1_ywaypoints[time]) ** 2)
            uuv_to_ob2_dist = math.sqrt((tmp.x - ob2_xwaypoints[time]) ** 2 + (tmp.y - ob2_ywaypoints[time]) ** 2)
            uuv_to_ob3_dist = math.sqrt((tmp.x - ob3_xwaypoints[time]) ** 2 + (tmp.y - ob3_ywaypoints[time]) ** 2)
            uuv_to_ob4_dist = math.sqrt((tmp.x - ob4_xwaypoints[time]) ** 2 + (tmp.y - ob4_ywaypoints[time]) ** 2)
            uuv_to_ob5_dist = math.sqrt((tmp.x - ob5_xwaypoints[time]) ** 2 + (tmp.y - ob5_ywaypoints[time]) ** 2)
            
            plt.cla()
            global ax
            cmap = colors.ListedColormap(['lightgrey', 'steelblue', 'lightgrey'])
            ax.imshow(worldmap, cmap=cmap, origin='lower')
            ax.scatter(startNode[0], startNode[1], marker="X", color="red", s=200)
            ax.scatter(goalNode[0], goalNode[1], marker="X", color="blue", s=200)
            ax.scatter(ox_2, oy_2, marker=".", color="black", s=50)
            ax.annotate("  Start", (startNode[0], startNode[1]))
            ax.annotate("  Goal  ", (goalNode[0], goalNode[1]))
            ax.xaxis.set_ticks(np.arange(0, 60, 1))
            plt.setp(ax.get_xticklabels(), rotation=90, ha='center')
            ax.yaxis.set_ticks(np.arange(0, 60, 1))
            plt.plot(dstar_x, dstar_y, color="yellow")
            plt.plot(GF_xc, GF_yc, color="red")
            plt.plot(MGF_xc, MGF_yc, color="green")
            uuv_x_coords = []
            uuv_y_coords = []
            for i in range(0, len(uuv_path)):
                uuv_x_coords.append(uuv_path[i][0] / 20)
                uuv_y_coords.append(uuv_path[i][1] / 20)
            plt.plot(uuv_x_coords, uuv_y_coords, color="magenta")
            patch1 = patches.Circle([ob1_xwaypoints[time], ob1_ywaypoints[time]], radius=0.25, color='b')
            ax.add_patch(patch1)
            plt.text(ob1_xwaypoints[time], ob1_ywaypoints[time], 'ob1')
            patch2 = patches.Circle([ob2_xwaypoints[time], ob2_ywaypoints[time]], radius=0.25, color='b')
            ax.add_patch(patch2)
            plt.text(ob2_xwaypoints[time], ob2_ywaypoints[time], 'ob2')
            patch3 = patches.Circle([ob3_xwaypoints[time], ob3_ywaypoints[time]], radius=0.25, color='b')
            ax.add_patch(patch3)
            plt.text(ob3_xwaypoints[time], ob3_ywaypoints[time], 'ob3')
            patch4 = patches.Circle([ob4_xwaypoints[time], ob4_ywaypoints[time]], radius=0.25, color='b')
            ax.add_patch(patch4)
            plt.text(ob4_xwaypoints[time], ob4_ywaypoints[time], 'ob4')
            patch5 = patches.Circle([ob5_xwaypoints[time], ob5_ywaypoints[time]], radius=0.25, color='b')
            ax.add_patch(patch5)
            plt.text(ob5_xwaypoints[time], ob5_ywaypoints[time], 'ob5')
            patchV = patches.Circle([tmp.x, tmp.y], radius=0.25, color='m')
            ax.add_patch(patchV)
            plt.text(tmp.x, tmp.y, 'usv')

            if uuv_to_ob1_dist < min_fixed_distance:
                print("Ob_1 At time: ", time)
                self.map.set_obstacle([(int(ob1_xwaypoints[time]-1), int(ob1_ywaypoints[time]-1))])
                print(int(ob1_xwaypoints[time]-1), int(ob1_ywaypoints[time]-1))
                self.map.set_obstacle([(int(ob1_xwaypoints[time]), int(ob1_ywaypoints[time]))])
                print(int(ob1_xwaypoints[time]), int(ob1_ywaypoints[time]))
                self.map.set_obstacle([(int(ob1_xwaypoints[time]+1), int(ob1_ywaypoints[time]+1))])
                print(int(ob1_xwaypoints[time]+1), int(ob1_ywaypoints[time]+1))
                self.modify(tmp)  # re-planning (tmp is the current uuv position)

            if uuv_to_ob2_dist < min_fixed_distance:
                print("Ob_2 At time: ", time)
                self.map.set_obstacle([(int(ob2_xwaypoints[time]-1), int(ob2_ywaypoints[time]-1))])
                print(int(ob2_xwaypoints[time]-1), int(ob2_ywaypoints[time]-1))
                self.map.set_obstacle([(int(ob2_xwaypoints[time]), int(ob2_ywaypoints[time]))])
                print(int(ob2_xwaypoints[time]), int(ob2_ywaypoints[time]))
                self.map.set_obstacle([(int(ob2_xwaypoints[time]+1), int(ob2_ywaypoints[time]+1))])
                print(int(ob2_xwaypoints[time]+1), int(ob2_ywaypoints[time]+1))
                self.modify(tmp)  # re-planning (tmp is the current uuv position)
                
            if uuv_to_ob3_dist < min_fixed_distance:
                print("Ob_3 At time: ", time)
                self.map.set_obstacle([(int(ob3_xwaypoints[time]), int(ob3_ywaypoints[time]-1))])
                print(int(ob3_xwaypoints[time]), int(ob3_ywaypoints[time]-1))
                self.map.set_obstacle([(int(ob3_xwaypoints[time]), int(ob3_ywaypoints[time]))])
                print(int(ob3_xwaypoints[time]), int(ob3_ywaypoints[time]))
                self.map.set_obstacle([(int(ob3_xwaypoints[time]+1), int(ob3_ywaypoints[time]+1))])
                print(int(ob3_xwaypoints[time]+1), int(ob3_ywaypoints[time]+1))
                self.modify(tmp)  # re-planning (tmp is the current uuv position)
                
            if uuv_to_ob4_dist < min_fixed_distance:
                print("Ob_4 At time: ", time)
                self.map.set_obstacle([(int(ob4_xwaypoints[time]+1), int(ob4_ywaypoints[time]-1))])
                print(int(ob4_xwaypoints[time]+1), int(ob4_ywaypoints[time]-1))
                self.map.set_obstacle([(int(ob4_xwaypoints[time]+2), int(ob4_ywaypoints[time]))])
                print(int(ob4_xwaypoints[time]+2), int(ob4_ywaypoints[time]))
                self.map.set_obstacle([(int(ob4_xwaypoints[time]+3), int(ob4_ywaypoints[time]+1))])
                print(int(ob4_xwaypoints[time]+3), int(ob4_ywaypoints[time]+1))
                self.modify(tmp)  # re-planning (tmp is the current uuv position)

            if uuv_to_ob5_dist < min_fixed_distance:
                print("Ob_5 At time: ", time)
                self.map.set_obstacle([(int(ob5_xwaypoints[time]), int(ob5_ywaypoints[time]))])
                print(int(ob5_xwaypoints[time]), int(ob5_ywaypoints[time]))
                self.map.set_obstacle([(int(ob5_xwaypoints[time]), int(ob5_ywaypoints[time]+1))])
                print(int(ob5_xwaypoints[time]), int(ob5_ywaypoints[time]+1))
                self.modify(tmp)  # re-planning (tmp is the current uuv position)
            
      #      if tmp.parent.state == "#":   # obstacle state
      #          self.modify(tmp)  # re-planning (tmp is the current uuv position)
      #          continue
            
            tmp = tmp.parent
            time = time+1
            plt.legend(['D*', 'Grassfire', 'MGrassfire', "USV"], loc=1)
            ax = plt.gca()
            leg = ax.get_legend()
            leg.legendHandles[0].set_color('yellow')
            leg.legendHandles[1].set_color('red')
            leg.legendHandles[2].set_color('green')
            leg.legendHandles[3].set_color('magenta')

            plt.grid(True)
            plt.axis("equal")
            plt.pause(0.01)
            
        tmp.set_state("e")  # oparent of current state
        time=0
        return dstar_x, dstar_y

            
############################### D Star End ######################################3

########## Run the UUV, Start ##########
fig, ax = plt.subplots(figsize=(8, 8))
uuv_path = global_path.copy()
time = 0
uuv_x = uuv_path[0][0]
uuv_y = uuv_path[0][1]
okayToCallLocalPlannerForOb1 = True
okayToCallLocalPlannerForOb2 = True
okayToCallLocalPlannerForOb3 = True
okayToCallLocalPlannerForOb4 = True
okayToCallLocalPlannerForOb5 = True
LocalPlannerCalledAtTime = math.inf
target1x = 0
target2x = 0
target3x = 0
target4x = 0
target5x = 0
target1y = 0
target2y = 0
target3y = 0
target4y = 0
target5y = 0

m = Map(60, 60)
m.set_obstacle([(i, j) for i, j in zip(ox, oy)])
start = m.map[startNode[0]][startNode[1]]
end = m.map[goalNode[0]][goalNode[1]]
dstar = Dstar(m)
dstar_x, dstar_y = dstar.run(start, end)
dstar_x.append(goalNode[0])
dstar_y.append(goalNode[1])
print("[(dstar_x, dstar_y)]: ", [(xc, yc) for xc, yc in zip(dstar_x, dstar_y)])

while (uuv_x / 20, uuv_y / 20) != (float(goalNode[0]), float(goalNode[1])):
    # dividing by 20 to get back the original 15 x 15 grid dimensions
    print("time = ", time)
    uuv_x = uuv_path[time][0]
    uuv_y = uuv_path[time][1]
    uuv_x_previous = uuv_path[time - 1][0]
    uuv_y_previous = uuv_path[time - 1][1]
    # Note: This use of previous assumes that there is no moving obstacle in close proximity of the start node, as
    # the previous point when time=0 is the last point in the path, namely the goal.
    uuv_to_ob1_dist = math.sqrt((uuv_x / 20 - ob1_xwaypoints[time]) ** 2 + (uuv_y / 20 - ob1_ywaypoints[time]) ** 2)
    uuv_to_ob2_dist = math.sqrt((uuv_x / 20 - ob2_xwaypoints[time]) ** 2 + (uuv_y / 20 - ob2_ywaypoints[time]) ** 2)
    uuv_to_ob3_dist = math.sqrt((uuv_x / 20 - ob3_xwaypoints[time]) ** 2 + (uuv_y / 20 - ob3_ywaypoints[time]) ** 2)
    uuv_to_ob4_dist = math.sqrt((uuv_x / 20 - ob4_xwaypoints[time]) ** 2 + (uuv_y / 20 - ob4_ywaypoints[time]) ** 2)
    uuv_to_ob5_dist = math.sqrt((uuv_x / 20 - ob5_xwaypoints[time]) ** 2 + (uuv_y / 20 - ob5_ywaypoints[time]) ** 2)

    if uuv_to_ob1_dist < min_fixed_distance:
        print("UUV is within close proximity of Obstacle 1")
        if okayToCallLocalPlannerForOb1:
            print("call local planner")
            ob_x = ob1_xwaypoints[time]
            ob_y = ob1_ywaypoints[time]
            ob_x_previous = ob1_xwaypoints[time - 5]
            ob_y_previous = ob1_ywaypoints[time - 5]
            uuv_path, tx, ty = local_path_planner(uuv_x, uuv_y, uuv_x_previous, uuv_y_previous,
                                                  ob_x, ob_y, ob_x_previous, ob_y_previous,
                                                  uuv_path)

            target1x = tx
            target1y = ty
            okayToCallLocalPlannerForOb1 = False
        else:
            if (target1x, target1y) != (uuv_x / 20, uuv_y / 20):
                print("waiting")
            else:
                okayToCallLocalPlannerForOb1 = True

    if uuv_to_ob2_dist < min_fixed_distance:
        print("UUV is within close proximity of Obstacle 2")
        if okayToCallLocalPlannerForOb2:
            print("call local planner")
            ob_x = ob2_xwaypoints[time]
            ob_y = ob2_ywaypoints[time]
            ob_x_previous = ob2_xwaypoints[time - 5]
            ob_y_previous = ob2_ywaypoints[time - 5]
            uuv_path, tx, ty = local_path_planner(uuv_x, uuv_y, uuv_x_previous, uuv_y_previous,
                                                  ob_x, ob_y, ob_x_previous, ob_y_previous,
                                                  uuv_path)
            target2x = tx
            target2y = ty
            okayToCallLocalPlannerForOb2 = False
        else:
            if (target2x, target2y) != (uuv_x / 20, uuv_y / 20):
                if uuv_to_ob2_dist < (min_fixed_distance / 7):
                    okayToCallLocalPlannerForOb2 = True
                else:
                    print("waiting")
            else:
                okayToCallLocalPlannerForOb2 = True

    if uuv_to_ob3_dist < min_fixed_distance:
        print("UUV is within close proximity of Obstacle 3")
        if okayToCallLocalPlannerForOb3:
            print("call local planner")
            ob_x = ob3_xwaypoints[time]
            ob_y = ob3_ywaypoints[time]
            ob_x_previous = ob3_xwaypoints[time - 5]
            ob_y_previous = ob3_ywaypoints[time - 5]
            uuv_path, tx, ty = local_path_planner(uuv_x, uuv_y, uuv_x_previous, uuv_y_previous,
                                                  ob_x, ob_y, ob_x_previous, ob_y_previous,
                                                  uuv_path)
            target3x = tx
            target3y = ty
            okayToCallLocalPlannerForOb3 = False
        else:
            if (target3x, target3y) != (uuv_x / 20, uuv_y / 20):
                if uuv_to_ob3_dist < (min_fixed_distance / 7):
                    okayToCallLocalPlannerForOb3 = True
                else:
                    print("waiting")
            else:
                okayToCallLocalPlannerForOb3 = True

    if uuv_to_ob4_dist < min_fixed_distance:
        print("UUV is within close proximity of Obstacle 4")
        if okayToCallLocalPlannerForOb4:
            print("call local planner")
            ob_x = ob4_xwaypoints[time]
            ob_y = ob4_ywaypoints[time]
            ob_x_previous = ob4_xwaypoints[time - 5]
            ob_y_previous = ob4_ywaypoints[time - 5]
            uuv_path, tx, ty = local_path_planner(uuv_x, uuv_y, uuv_x_previous, uuv_y_previous,
                                                  ob_x, ob_y, ob_x_previous, ob_y_previous,
                                                  uuv_path)
            target4x = tx
            target4y = ty
            okayToCallLocalPlannerForOb4 = False
        else:
            if (target4x, target4y) != (uuv_x / 20, uuv_y / 20):
                if uuv_to_ob4_dist < (min_fixed_distance / 7):
                    okayToCallLocalPlannerForOb4 = True
                else:
                    print("waiting")
            else:
                okayToCallLocalPlannerForOb4 = True

    if uuv_to_ob5_dist < min_fixed_distance:
        print("UUV is within close proximity of Obstacle 5")
        if okayToCallLocalPlannerForOb5:
            print("call local planner")
            ob_x = ob5_xwaypoints[time]
            ob_y = ob5_ywaypoints[time]
            ob_x_previous = ob5_xwaypoints[time - 5]
            ob_y_previous = ob5_ywaypoints[time - 5]
            uuv_path, tx, ty = local_path_planner(uuv_x, uuv_y, uuv_x_previous, uuv_y_previous,
                                                  ob_x, ob_y, ob_x_previous, ob_y_previous,
                                                  uuv_path)
            target5x = tx
            target5y = ty
            okayToCallLocalPlannerForOb5 = False
        else:
            if (target5x, target5y) != (uuv_x / 20, uuv_y / 20):
                print("waiting")
            else:
                okayToCallLocalPlannerForOb5 = True

    time = time + 1
########## Run the UUV, End ##########

########## Output the map, Start ##########
    plt.cla()
    cmap = colors.ListedColormap(['lightgrey', 'steelblue', 'lightgrey'])
    ax.imshow(worldmap, cmap=cmap, origin='lower')
    ax.scatter(startNode[0], startNode[1], marker="X", color="red", s=200)
    ax.scatter(goalNode[0], goalNode[1], marker="X", color="blue", s=200)
    ax.scatter(ox_2, oy_2, marker=".", color="black", s=50)
    ax.annotate("  Start", (startNode[0], startNode[1]))
    ax.annotate("  Goal  ", (goalNode[0], goalNode[1]))
    ax.xaxis.set_ticks(np.arange(0, 60, 1))
    plt.setp(ax.get_xticklabels(), rotation=90, ha='center')
    ax.yaxis.set_ticks(np.arange(0, 60, 1))
    plt.plot(dstar_x, dstar_y, color="yellow")
    plt.plot(GF_xc, GF_yc, color="red")
    plt.plot(MGF_xc, MGF_yc, color="green")
    uuv_x_coords = []
    uuv_y_coords = []
    for i in range(0, len(uuv_path)):
        uuv_x_coords.append(uuv_path[i][0] / 20)
        uuv_y_coords.append(uuv_path[i][1] / 20)
    plt.plot(uuv_x_coords, uuv_y_coords, color="magenta")
    patch1 = patches.Circle([ob1_xwaypoints[time], ob1_ywaypoints[time]], radius=0.25, color='b')
    ax.add_patch(patch1)
    plt.text(ob1_xwaypoints[time], ob1_ywaypoints[time], 'ob1')
    patch2 = patches.Circle([ob2_xwaypoints[time], ob2_ywaypoints[time]], radius=0.25, color='b')
    ax.add_patch(patch2)
    plt.text(ob2_xwaypoints[time], ob2_ywaypoints[time], 'ob2')
    patch3 = patches.Circle([ob3_xwaypoints[time], ob3_ywaypoints[time]], radius=0.25, color='b')
    ax.add_patch(patch3)
    plt.text(ob3_xwaypoints[time], ob3_ywaypoints[time], 'ob3')
    patch4 = patches.Circle([ob4_xwaypoints[time], ob4_ywaypoints[time]], radius=0.25, color='b')
    ax.add_patch(patch4)
    plt.text(ob4_xwaypoints[time], ob4_ywaypoints[time], 'ob4')
    patch5 = patches.Circle([ob5_xwaypoints[time], ob5_ywaypoints[time]], radius=0.25, color='b')
    ax.add_patch(patch5)
    plt.text(ob5_xwaypoints[time], ob5_ywaypoints[time], 'ob5')
    patchV = patches.Circle([uuv_x / 20, uuv_y / 20], radius=0.25, color='m')
    ax.add_patch(patchV)
    plt.text(uuv_x / 20, uuv_y / 20, 'usv')
    plt.legend(['D*', 'Grassfire', 'MGrassfire', "USV"], loc=1)
    ax = plt.gca()
    leg = ax.get_legend()
    leg.legendHandles[0].set_color('yellow')
    leg.legendHandles[1].set_color('red')
    leg.legendHandles[2].set_color('green')
    leg.legendHandles[3].set_color('magenta')

    plt.grid(True)
    plt.axis("equal")
    plt.pause(0.01)
    
plt.show()
####### Output the map, End ##########

print('Moving obstacle_1 speed is: {} m/s, yaw rate is: {} rad/s'.format(round(mob_1_speed, 2), round(yaw1, 2)))
print('Moving obstacle_2 speed is: {} m/s, yaw rate is: {} rad/s'.format(round(mob_2_speed, 2), round(yaw2, 2)))
print('Moving obstacle_3 speed is: {} m/s, yaw rate is: {} rad/s'.format(round(mob_3_speed, 2), round(yaw3, 2)))
print('Moving obstacle_4 speed is: {} m/s, yaw rate is: {} rad/s'.format(round(mob_4_speed, 2), round(yaw4, 2)))
print('Moving obstacle_5 speed is: {} m/s, yaw rate is: {} rad/s'.format(round(mob_5_speed, 2), round(yaw5, 2)))
print('Remus uuv speed is: {} m/s, yaw rate is: {} rad/s'.format(round(uuv_speed, 2), round(yawUUV, 2)))
