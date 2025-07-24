import gymnasium as gym
import math
import random
import numpy as np
import Box2D
import cv2

from typing import Optional

# Constants
WIDTH, HEIGHT = 1000, 600
PPM = 20.0
TARGET_FPS = 30
TIME_STEP = 1.0 / TARGET_FPS
GRAVITY = -30
DAMP_RATE = 1.15
JUMP_FORCE = 100

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (100, 200, 100)
BLUE = (50, 50, 255)
RED = (255, 50, 50)
YELLOW = (255, 255, 0)

# Goals
GOAL_WIDTH = 1
GOAL_HEIGHT = 7
CROSSBAR_WIDTH = 4
CROSSBAR_HEIGHT = 1
LEFT_GOAL_X = 1
RIGHT_GOAL_X = 49
KICK_STRENGTH = 60

class SoccerContactListener(Box2D.b2ContactListener):
    def __init__(self, jump_count, goal, **kwargs):
        self.jump_count = jump_count
        self.goal = goal
        super().__init__(**kwargs)
    
    def BeginContact(self, contact):
        a = contact.fixtureA
        b = contact.fixtureB
        labels = {a.userData, b.userData, a.body.userData, b.body.userData}

        # Reset jump count if player touches ground
        for label in ["Player1", "Player2"]:
            if label in labels and "Ground" in labels:
                self.jump_count[label] = 0

        for label in ["LeftNet", "RightNet"]:
            if label in labels and "Ball" in labels:
                self.goal(label)

def to_numpy(v):
    return np.array([v.x, v.y], dtype=np.float32)

class SoccerEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode = "human"):
        self.render_mode = render_mode
        
        # variables
        self.jump_count = {"Player1": 0, "Player2": 0}
        self.invalidate_jump = {"Player1": 0, "Player2": 0}
        self.terminated = False
        self.winner = None
        
        # Box2D world (gravity downwards)
        world = Box2D.b2World(gravity=(0, GRAVITY), doSleep=True)
        self.world = world
        world.contactListener = SoccerContactListener(self.jump_count, self._goal)
        
        # Field and ground
        ground = world.CreateStaticBody(position=(25, 1), userData = "Ground")
        ground.CreatePolygonFixture(box=(25, 1), friction = 0.3)
        
        # Side walls (left/right)
        world.CreateStaticBody(position=(0, 15)).CreatePolygonFixture(box=(1, 15))
        world.CreateStaticBody(position=(50, 15)).CreatePolygonFixture(box=(1, 15))
        
        # Net
        left_post = world.CreateStaticBody(position=(LEFT_GOAL_X, GOAL_HEIGHT / 2 + 1), userData = "LeftNet")
        left_post.CreatePolygonFixture(box=(0.2, GOAL_HEIGHT / 2))

        left_crossbar = world.CreateStaticBody(position=(LEFT_GOAL_X + CROSSBAR_WIDTH / 2, GOAL_HEIGHT + CROSSBAR_HEIGHT))
        left_crossbar.CreatePolygonFixture(box=(CROSSBAR_WIDTH / 2, CROSSBAR_HEIGHT / 2))

        right_post = world.CreateStaticBody(position=(RIGHT_GOAL_X, GOAL_HEIGHT / 2 + 1), userData = "RightNet")
        right_post.CreatePolygonFixture(box=(0.2, GOAL_HEIGHT / 2))

        right_crossbar = world.CreateStaticBody(position=(RIGHT_GOAL_X - CROSSBAR_WIDTH / 2, GOAL_HEIGHT + CROSSBAR_HEIGHT))
        right_crossbar.CreatePolygonFixture(box=(CROSSBAR_WIDTH / 2, CROSSBAR_HEIGHT / 2))

        # Ball
        ball = world.CreateDynamicBody(position=(25, 5), bullet=True, linearDamping = DAMP_RATE, userData = "Ball")
        self.ball = ball
        ball.CreateCircleFixture(radius=0.5, density=1, friction=0.3, restitution=0.8)
        
        # Players
        def create_player(x, y, label: str):
            body = world.CreateDynamicBody(position=(x, y), fixedRotation=True,
                                           linearDamping = DAMP_RATE)
            body.CreateCircleFixture(radius=1, density=2, friction=0.1, restitution=0.3,userData = label)
            return body

        self.player1 = create_player(10, 2, "Player1")
        self.player2 = create_player(40, 2, "Player2")
        
        self.movement_force = 80
        
        self.max_speed = 1000 # temp

        self.observation_space = gym.spaces.Dict(
            {
                "Player1": gym.spaces.Box(low=0, high=50, shape=(2,), dtype=np.float32),
                "Player1Velocity": gym.spaces.Box(low=-self.max_speed, high=self.max_speed, shape=(2,), dtype=np.float32),
                "Player2": gym.spaces.Box(low=0, high=50, shape=(2,), dtype=np.float32),
                "Player2Velocity": gym.spaces.Box(low=-self.max_speed, high=self.max_speed, shape=(2,), dtype=np.float32),
                "Ball": gym.spaces.Box(low=0, high=50, shape=(2,), dtype=np.float32),
                "BallVelocity": gym.spaces.Box(low=-self.max_speed, high=self.max_speed, shape=(2,), dtype=np.float32),
            }
        )

        self.action_space = gym.spaces.MultiDiscrete([5, 5])
        # 0: idle
        # 1: move left
        # 2: move right
        # 3: jump
        # 4: kick
        
    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """
        
        obs = {
            "Player1": to_numpy(self.player1.position),
            "Player1Velocity": to_numpy(self.player1.linearVelocity),
            "Player2": to_numpy(self.player2.position),
            "Player2Velocity": to_numpy(self.player2.linearVelocity),
            "Ball": to_numpy(self.ball.position),
            "BallVelocity": to_numpy(self.ball.linearVelocity),
        }
        
        return obs
    
    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        return {
            "winner": self.winner
        }
    
    def _act(self, player, label, action):
        if action == 1:
            player.ApplyForceToCenter((-self.movement_force, 0), True)
        elif action == 2:
            player.ApplyForceToCenter((self.movement_force, 0), True)
        elif action == 3:
            if self.jump_count[label] < 1 and self.invalidate_jump[label] < 0:
                player.ApplyLinearImpulse((0, JUMP_FORCE), player.worldCenter, True)
                self.jump_count[label] = 1
                self.invalidate_jump[label] = 3
        elif action == 4:
            self._try_kick(player)
            
    def _try_kick(self, player):
        dist = self.ball.position - player.position
        if dist.length >= 2:
            return
        
        direction = self.ball.position - player.position
        if direction.length == 0:
            return
        direction.Normalize()
        self.ball.ApplyLinearImpulse((direction.x * KICK_STRENGTH, direction.y * KICK_STRENGTH), self.ball.worldCenter, True)
    
    def _goal(self, net_name):
        self.terminated = True
        if net_name == "LeftNet":
            self.winner = "Player2"
        elif net_name == "RightNet":
            self.winner = "Player1"
    
    def _to_screen(self, x, y):
        return int(x * PPM), int(HEIGHT - y * PPM)
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        self.ball.position = (25, 3)
        self.ball.linearVelocity = (0, 0)
        self.player1.position = (self.np_random.random() * 44 + 3, 3)
        self.player1.linearVelocity = (0, 0)
        self.player2.position = (50 - self.player1.position.x, 3)
        self.player2.linearVelocity = (0, 0)
        
        self.jump_count["Player1"] = 0
        self.jump_count["Player2"] = 0
        self.invalidate_jump["Player1"] = 0
        self.invalidate_jump["Player2"] = 0
        self.terminated = False
        self.winner = None

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        
        for i, player, label in [(0, self.player1, "Player1"), (1, self.player2, "Player2")]:
            self._act(player, label, action[i])
        
        self.world.Step(TIME_STEP, 10, 10)
        
        for key in self.invalidate_jump:
            self.invalidate_jump[key] -= 1
        
        # truncate will be implemented by gym.register
        truncated = False
        
        if self.player1.linearVelocity.y > 15.:
            self.player1.linearVelocity.y = 15
        if self.player2.linearVelocity.y > 15.:
            self.player2.linearVelocity.y = 15

        reward = 0 # gymnesium은 2차원 reward를 허용하지 않음. info로 대체.

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, self.terminated, truncated, info
    
    def render(self):
        if self.render_mode != "rgb_array":
            return
        
        frame = np.full((HEIGHT, WIDTH, 3), GREEN, dtype=np.uint8)
        
        # Draw ground
        cv2.rectangle(frame,
                      (0, int(HEIGHT - 2 * PPM)),
                      (int(WIDTH), int(HEIGHT)),
                      BLACK,
                      -1)

        # Draw goal posts
        cv2.rectangle(frame,
                      self._to_screen(LEFT_GOAL_X, 2 + GOAL_HEIGHT),
                      (self._to_screen(LEFT_GOAL_X, 2 + GOAL_HEIGHT)[0] + 5,
                       self._to_screen(LEFT_GOAL_X, 2 + GOAL_HEIGHT)[1] + int(GOAL_HEIGHT * PPM)),
                      YELLOW,
                      -1)
        cv2.rectangle(frame,
                      self._to_screen(RIGHT_GOAL_X, 2 + GOAL_HEIGHT),
                      (self._to_screen(RIGHT_GOAL_X, 2 + GOAL_HEIGHT)[0] + 5,
                       self._to_screen(RIGHT_GOAL_X, 2 + GOAL_HEIGHT)[1] + int(GOAL_HEIGHT * PPM)),
                      YELLOW,
                      -1)

        # Draw crossbars
        cv2.rectangle(frame,
                      self._to_screen(LEFT_GOAL_X, GOAL_HEIGHT + 2),
                      (self._to_screen(LEFT_GOAL_X, GOAL_HEIGHT + 2)[0] + int(CROSSBAR_WIDTH * PPM),
                       self._to_screen(LEFT_GOAL_X, GOAL_HEIGHT + 2)[1] + int(CROSSBAR_HEIGHT * PPM)),
                      WHITE,
                      -1)
        cv2.rectangle(frame,
                      self._to_screen(RIGHT_GOAL_X - CROSSBAR_WIDTH, GOAL_HEIGHT + 2),
                      (self._to_screen(RIGHT_GOAL_X - CROSSBAR_WIDTH, GOAL_HEIGHT + 2)[0] + int(CROSSBAR_WIDTH * PPM),
                       self._to_screen(RIGHT_GOAL_X - CROSSBAR_WIDTH, GOAL_HEIGHT + 2)[1] + int(CROSSBAR_HEIGHT * PPM)),
                      WHITE,
                      -1)

        # Draw ball
        cv2.circle(frame,
                   self._to_screen(*self.ball.position),
                   int(0.5 * PPM),
                   WHITE,
                   -1)

        # Draw players
        cv2.circle(frame,
                   self._to_screen(*self.player1.position),
                   int(1 * PPM),
                   BLUE,
                   -1)
        cv2.circle(frame,
                   self._to_screen(*self.player2.position),
                   int(1 * PPM),
                   RED,
                   -1)
        
        return frame

gym.register(
    id="SoccerEnv",
    entry_point=SoccerEnv,
    max_episode_steps=100,  # Prevent infinite episodes
)

if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env
    import traceback

    env = gym.make("SoccerEnv")

    # This will catch many common issues
    try:
        check_env(env.unwrapped)
        print("Environment passes all checks!")
    except Exception as e:
        print(f"Environment has issues: {e}")
        traceback.print_exc()