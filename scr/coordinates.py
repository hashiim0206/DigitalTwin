import sumolib

NET_FILE = "version.net.xml"
net = sumolib.net.readNet(NET_FILE)

# Major directions and the edges we care about
MAIN_DIRS = {
    "W": ["W", "W_0", "W_1", "W_2", "W_3", "W_4"],
    "E": ["E", "E_0", "E_1", "E_2", "E_3", "E_4"],
    "N": ["N", "N_0", "N_1", "N_2", "N_3", "N_4"],
    "S": ["S", "S_0", "S_1", "S_2", "S_3", "S_4"],
}

def get_extreme_point(edges):
    """Find the farthest point on a group of lane shapes."""
    pts = []
    for edge_id in edges:
        edge = net.getEdge(edge_id)
        if not edge:
            continue
        for lane in edge.getLanes():
            pts.extend(lane.getShape())
    return pts

# ---- A. Outer road endpoints ----
west_pts  = get_extreme_point(MAIN_DIRS["W"])
east_pts  = get_extreme_point(MAIN_DIRS["E"])
north_pts = get_extreme_point(MAIN_DIRS["N"])
south_pts = get_extreme_point(MAIN_DIRS["S"])

west_end  = min(west_pts,  key=lambda p: p[0])
east_end  = max(east_pts,  key=lambda p: p[0])
north_end = max(north_pts, key=lambda p: p[1])
south_end = min(south_pts, key=lambda p: p[1])

# ---- B. Intersection bounding corners ----
xmin = min(p[0] for p in north_pts + south_pts + west_pts + east_pts)
xmax = max(p[0] for p in north_pts + south_pts + west_pts + east_pts)
ymin = min(p[1] for p in north_pts + south_pts + west_pts + east_pts)
ymax = max(p[1] for p in north_pts + south_pts + west_pts + east_pts)

# approximate corners
intersection_top_left     = (xmin, ymax)
intersection_top_right    = (xmax, ymax)
intersection_bottom_left  = (xmin, ymin)
intersection_bottom_right = (xmax, ymin)

# ---- C. Reference lane: E_through ----
# Replace with your actual lane name from net.xml!
# Example: "E_2" or internal lane names like "E_0_0"
ref_lane = net.getLane("E_0")
ref_points = ref_lane.getShape()

ref_start = ref_points[0]
ref_end   = ref_points[-1]

# ---- PRINT EVERYTHING ----
print("\n===== Extracted Coordinates =====")
print("West road end:", west_end)
print("East road end:", east_end)
print("North road end:", north_end)
print("South road end:", south_end)

print("\nIntersection corners:")
print("Top-left:", intersection_top_left)
print("Top-right:", intersection_top_right)
print("Bottom-left:", intersection_bottom_left)
print("Bottom-right:", intersection_bottom_right)

print("\nE_through reference:")
print("Start:", ref_start)
print("End:", ref_end)
