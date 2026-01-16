# 경로생성 = D*lite / 주행 = pure pursuit 

"""
ros2 launch turtlebot3_bringup robot.launch.py

ros2 launch nav2_bringup localization_launch.py map:=/내/맵파일/절대경로/map.yaml

ros2 run rviz2 rviz2 -d /home/jgs/patrol_pkg/patrol_pkg/patrol.rviz

"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener
from rclpy.duration import Duration
from math import hypot, atan2, cos, sin, pi
import heapq
import numpy as np

# =========================================
# 1. D* Lite (경로 생성)
# =========================================
class DStarLite:
    def __init__(self):
        self.width = 0; self.height = 0
        self.map = None
        self.g = {}; self.rhs = {}; self.U = []        # g : 목표한 지점까지의 실제 비용 , rhs : 예측 비용(lookahead)
        self.km = 0; self.start = None; self.goal = None
        self.resolution = 0.05
        self.inflation_radius = 0 # 안전반경 초기화 

    # 맵 데이터를 받아 초기화 및 장애물 inflation 설정 
    def init_map(self, width, height, data, resolution, inflation_radius_cells):
        self.width = width; self.height = height; self.resolution = resolution
        self.inflation_radius = inflation_radius_cells
        
        # 1차원 맵 -> 2차원 데이터 변환 
        # 50 보다 크면 장애물(1) / 아니면 빈 공간(0)
        grid = np.array(data).reshape((height, width))
        self.map = np.where(grid > 50, 1, 0)
        
        # 장애물 inflation(부풀리기) 설정 
        self.inflate_obstacles()

        # 경로 계산 관련 변수 초기화 
        self.g.clear(); self.rhs.clear(); self.U = []
        self.km = 0

    # inflation 설정 로직 (장애물 확장)
    def inflate_obstacles(self):
        inflated_map = np.copy(self.map)     # 원본 맵 복사 
        rows, cols = np.where(self.map == 1) # 실제 장애물 위치 찾기
        
        for y, x in zip(rows, cols):
            # 장애물 주변을 inflation_radius 크기만큼 확장 
            for dy in range(-self.inflation_radius, self.inflation_radius + 1):
                for dx in range(-self.inflation_radius, self.inflation_radius + 1):
                    ny, nx = y + dy, x + dx
                    # 맵 범위를 벗어나는지 체크 
                    if 0 <= ny < self.height and 0 <= nx < self.width:
                        # 설정한 반경 안에 있으면 장애물로 설정 
                        if hypot(dx, dy) <= self.inflation_radius:
                            inflated_map[ny][nx] = 1
        self.map = inflated_map
    
    # 휴리스틱 함수 (남은거리 추정)
    # 대각선 이동을 고려한 거리 계산 (Octile Distance)
    def heuristic(self, a, b):
        dx = abs(a[0] - b[0]); dy = abs(a[1] - b[1])
        return 1.414 * min(dx, dy) + (max(dx, dy) - min(dx, dy))

    # 우선순위 큐(Heapq)용 키 계산 
    # [f-cost, g-cost] 형태. f-cost가 낮을수록 우선순위 높음 
    def calculate_key(self, s):
        m = min(self.g.get(s, float('inf')), self.rhs.get(s, float('inf')))
        return [m + self.heuristic(self.start, s) + self.km, m]

    # 경로 탐색 초기화 및 목표 설정 
    def initialize(self, start, goal):
        self.start = start; self.goal = goal
        self.g.clear(); self.rhs.clear(); self.U = []
        self.km = 0
        self.rhs[self.goal] = 0 # 목표점 비용 초기값 0 설정 
        # 목표설정 -> heapq에 넣어 탐색 (D* Lite는 목표에서 시작점으로 역탐색)
        heapq.heappush(self.U, self.calculate_key(self.goal) + list(self.goal))

    # 노드 상태 업데이트 
    def update_vertex(self, u):
        if u != self.goal:
            # 주변 이웃 중 최소 비용을 찾아 rhs(예측비용) 갱신
            self.rhs[u] = min(self.g.get(n, float('inf')) + self.cost(u, n) for n in self.get_neighbors(u))
        # heapq에서 기존 항목 제거 후 갱신된 값으로 재삽입
        self.U = [i for i in self.U if (i[2], i[3]) != u]
        heapq.heapify(self.U)
        if self.g.get(u, float('inf')) != self.rhs.get(u, float('inf')):
            heapq.heappush(self.U, self.calculate_key(u) + list(u))

    # 최단 경로 계산 
    def compute_shortest_path(self):
        limit = 0
        # 시작점이 해결되거나 heapq가 빌 때까지 반복 -> 목적지 여러번 찍기 가능 
        while self.U and (self.U[0][:2] < self.calculate_key(self.start) or self.rhs.get(self.start, float('inf')) != self.g.get(self.start, float('inf'))):
            limit += 1
            if limit > 5000: break # 무한루프 방지
            u_item = heapq.heappop(self.U)
            u = (u_item[2], u_item[3])
            
            # g[u] : 현기준 최단거리 / rhs[u] : 주변 이웃들을 토대로 재계산 했을때의 최단거리(기대값) 
            # g > rhs 재계산한 거리가 더 낫다 
            if self.g.get(u, float('inf')) > self.rhs.get(u, float('inf')):
                self.g[u] = self.rhs[u]
                for s in self.get_neighbors(u): self.update_vertex(s)
            # g <= rhs  기존 최단거리가 낫다 -> 장애물 발견 
            else:
                self.g[u] = float('inf')
                self.update_vertex(u)
                for s in self.get_neighbors(u): self.update_vertex(s)

    # grid 격자 기준 이웃 노드 반환 (8방향 -> 상 하 좌 우 , 대각선 4방향)
    def get_neighbors(self, u):
        x, y = u
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height: yield (nx, ny)

    # 이동 비용 계산 (장애물은 무한대, 대각선 1.414(피타고라스 정리 의거), 직선 1.0)
    # 대각선 보단 직선 위주의 주행 우선순위 두기 
    def cost(self, u, v):
        if self.map[u[1]][u[0]] == 1 or self.map[v[1]][v[0]] == 1: return float('inf')
        return 1.414 if abs(u[0]-v[0]) + abs(u[1]-v[1]) == 2 else 1.0

    # 계산된 경로 역추적(D*lite) 하여 리스트로 반환 -> path 생성 
    def extract_path(self):
        if self.rhs.get(self.start, float('inf')) == float('inf'): return [] # 경로 없음
        path = [self.start]; curr = self.start
        for _ in range(5000):       # 무한루프 방지 
            if curr == self.goal: break
            # 가장 비용이 낮은 이웃을 따라감 (Gradient Descent) -> 우선순위 높은 곳 찾아가기 
            next_node = min(self.get_neighbors(curr), key=lambda n: self.cost(curr, n) + self.g.get(n, float('inf')), default=None)
            if next_node is None: break
            path.append(next_node); curr = next_node
        return path

# =========================================
# 주행(Pure Pursuit) 
# =========================================
class PatrolStaticNode(Node):
    def __init__(self):
        super().__init__('patrol_static')
        
        self.planner = DStarLite()
        self.origin_x = 0.0; self.origin_y = 0.0; self.resolution = 0.05
        self.full_path = []
        self.is_moving = False
        self.map_initialized = False 

        # TF(좌표변환) 
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # QoS 설정
        qos_map = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL, history=HistoryPolicy.KEEP_LAST, depth=1)
        qos_scan = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, depth=10)

        # 토픽 구독 및 발행
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, qos_map)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_scan)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10) # 로봇 속도 명령
        self.pub_path = self.create_publisher(Path, '/plan_path', 10) # 경로 시각화
        
        # timer 
        self.create_timer(0.1, self.control_loop)           # 0.1 
        self.get_logger().info("Patrol Node Ready. Waiting for Map...")

    # World 좌표(m) -> Grid 좌표(index) 변환 => 지도상의 로봇 위치와 데이터 그리기 위함
    def world_to_grid(self, wx, wy):
        return int((wx - self.origin_x) / self.resolution), int((wy - self.origin_y) / self.resolution)

    # Grid 좌표(index) -> World 좌표(m) 변환 => 지도를 통해 찾은 경로 주행명령 내리기 위함 
    def grid_to_world(self, gx, gy):
        return gx * self.resolution + self.origin_x, gy * self.resolution + self.origin_y

    # 현재 로봇 위치(x, y, yaw) 조회 함수 (TF 사용) 
    # -> 지도의 원점과 로봇의 중심점 까지의 거리, 회전각도를 계산하여 반환 
    def get_robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_footprint', rclpy.time.Time(), timeout=Duration(seconds=0.2))
            q = t.transform.rotation
            # 쿼터니언(q.w, q.z, q.x, q.y)을 라디안 각(Yaw)으로 변환
            yaw = atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
            return t.transform.translation.x, t.transform.translation.y, yaw
        except: return None

    # 맵 데이터 수신 콜백
    def map_callback(self, msg):
        self.resolution = msg.info.resolution
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y
        
        # 안전거리 offset inflation 설정
        inflation_cells = int(0.2 / self.resolution)

        # 플래너에 맵 데이터 전달 및 장애물 inflation 설정 
        self.planner.init_map(msg.info.width, msg.info.height, msg.data, self.resolution, inflation_cells)
        
        if not self.map_initialized:
            self.map_initialized = True
            self.get_logger().info(f"Map Loaded. Obstacles Inflated by {inflation_cells} cells.")

    # 목표 지점(2D Goal) 수신 콜백
    def goal_callback(self, msg):
        pose = self.get_robot_pose()
        if pose is None: return

        # 시작점과 목표점을 그리드 좌표로 변환
        start_grid = self.world_to_grid(pose[0], pose[1])
        goal_grid = self.world_to_grid(msg.pose.position.x, msg.pose.position.y)

        self.get_logger().info(f"New Goal -> ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})")
        
        # 시작점 주변 강제 초기화 -> 다시 goal을 찍을때 오류 방지 
        # 로봇이 벽 근처에서 출발할 때 자신을 장애물로 인식하여 경로 생성을 실패하는 것 방지
        sx, sy = start_grid
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                ny, nx = sy + dy, sx + dx
                if 0 <= ny < self.planner.height and 0 <= nx < self.planner.width:
                    # 시작점 주변 3x3 영역이 장애물(1)이라면 0으로 강제 할당 
                    if self.planner.map[ny][nx] == 1:
                        self.planner.map[ny][nx] = 0
                        self.planner.update_vertex((nx, ny)) # 변경된 정보 맵에 업데이트

        # 경로 계획 실행 (D* Lite)
        self.planner.initialize(start_grid, goal_grid)
        self.planner.compute_shortest_path()
        self.full_path = self.planner.extract_path()
        
        if self.full_path:
            self.is_moving = True
            self.publish_path(self.full_path) # Rviz에 경로 표시
        else:
            self.get_logger().warn("Path planning failed! (Start or Goal might be in obstacle)")

    # 동적 장애물 처리
    def scan_callback(self, msg):
        if not self.is_moving: return
        pose = self.get_robot_pose()
        if pose is None: return

        rx, ry, ryaw = pose
        for i, r in enumerate(msg.ranges):
            # 성능 최적화를 위해 데이터 다운샘플링 (4개 중 1개만 사용)
            if i % 4 != 0: continue

            # 장애물 유효 거리(0.1m ~ 2.0m)
            if r < 0.1 or r > 2.0: continue

            # 장애물의 절대 좌표 계산
            angle = msg.angle_min + i * msg.angle_increment + ryaw
            ox = rx + r * cos(angle)
            oy = ry + r * sin(angle)
            gx, gy = self.world_to_grid(ox, oy)

            # 맵 범위 내에 새로운 장애물이 발견되면 맵 업데이트
            if 0 <= gx < self.planner.width and 0 <= gy < self.planner.height:
                if self.planner.map[gy][gx] == 0:
                    self.planner.map[gy][gx] = 1
                    self.planner.update_vertex((gx, gy))

    # 주행 제어 (Pure Pursuit)
    def control_loop(self):
        if not self.full_path: return
        
        pose = self.get_robot_pose()
        if pose is None: return
        rx, ry, ryaw = pose



        # Pruning (지나온 길의 점들 삭제)
        # 로봇과 가장 가까운 경로 점을 찾아 그 이전 경로들을 리스트에서 제거
        min_dist = float('inf'); closest_idx = -1
        for i, (gx, gy) in enumerate(self.full_path):
            wx, wy = self.grid_to_world(gx, gy)
            dist = hypot(wx - rx, wy - ry)
            if dist < min_dist: min_dist = dist; closest_idx = i
        
        if closest_idx > 0: self.full_path = self.full_path[closest_idx:]



        # Lookahead (전방 주시 지점 선정)
        # 로봇으로부터 0.4m 떨어진 경로 점을 목표로 설정
        LOOKAHEAD_DIST = 0.6
        target_x, target_y = None, None
        for gx, gy in self.full_path:
            wx, wy = self.grid_to_world(gx, gy)
            if hypot(wx - rx, wy - ry) > LOOKAHEAD_DIST:
                target_x, target_y = wx, wy; break
        
        # Lookahead 지점이 없으면 경로의 마지막 점을 목표로 함
        if target_x is None: target_x, target_y = self.grid_to_world(*self.full_path[-1])

        # 목표 지점이 어디인가(거리와 각도 판단)
        dist_to_goal = hypot(target_x - rx, target_y - ry)
        angle_to_goal = atan2(target_y - ry, target_x - rx)
        # 얼마나 꺾어야 하는가(오차계산 -> 최단거리 각도 설정) 
        yaw_err = (angle_to_goal - ryaw + pi) % (2 * pi) - pi

        cmd = Twist()
        
        # 도착 판정 (남은 경로 점이 5개 미만이고 거리가 0.2m 이내일 때)     
        # -> 더 정밀하게 = 점 개수 줄이기 + 남은거리 0.1m
        # 지나온 길의 점은 삭제됨 -> 남은 경로 점 5개 미만 = 다 온거나 마찬가지 
        if len(self.full_path) < 5 and dist_to_goal < 0.1:
            self.get_logger().info("Goal Reached!")
            self.full_path = []
            self.is_moving = False
            self.pub_cmd.publish(Twist()) # 정지
            return

        # P-Controller (비례 제어)
        # yaw_err = 각도의 오차범위 
        cmd.angular.z = 0.5 * yaw_err # 회전 속도
        
        # 각도 오차가 작으면 빠르게, 크면 느리게 또는 제자리 회전
        if abs(yaw_err) < 0.5: cmd.linear.x = 0.15
        elif abs(yaw_err) < 1.0: cmd.linear.x = 0.1
        else: cmd.linear.x = 0.0

        self.pub_cmd.publish(cmd)

    # 경로 시각화(path_plan)
    def publish_path(self, nodes):
        msg = Path()
        msg.header.frame_id = 'map'
        for gx, gy in nodes:
            p = PoseStamped()
            p.pose.position.x, p.pose.position.y = self.grid_to_world(gx, gy)
            p.pose.orientation.w = 1.0
            msg.poses.append(p)
        self.pub_path.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = PatrolStaticNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()