import pygame
from Box2D import *
#from pygame_screen_record import ScreenRecorder

# Constants
WIDTH, HEIGHT = 1000, 600
PPM = 20.0
TARGET_FPS = 30
TIME_STEP = 1.0 / TARGET_FPS
GRAVITY = -30
DAMP_RATE = 1.15
JUMP_FORCE = 100
# IS_RECORDING = False

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (100, 200, 100)
BLUE = (50, 50, 255)
RED = (255, 50, 50)
YELLOW = (255, 255, 0)

# Variables
jump_count = {"Player1": 0, "Player2": 0}
# Goals
GOAL_WIDTH = 1
GOAL_HEIGHT = 7
CROSSBAR_WIDTH = 4
CROSSBAR_HEIGHT = 1
LEFT_GOAL_X = 1
RIGHT_GOAL_X = 49
KICK_STRENGTH = 60

winner: str = "None"

#아래 두 변수는 플레이어가 계속 땅과 충돌 판정이 나서 비정상적인 점프력을 얻는 오류를 해결하기 위해 도입했다.
player1_invalidate_jump = 0
player2_invalidate_jump = 0
timestep = 0
terminated = False

class MyContactListener(b2ContactListener):
    def BeginContact(self, contact):
        a = contact.fixtureA
        b = contact.fixtureB
        labels = {a.userData, b.userData, a.body.userData, b.body.userData}


        # Reset jump count if player touches ground
        for label in ["Player1", "Player2"]:
            if label in labels and "Ground" in labels:
                jump_count[label] = 0

        for label in ["LeftNet", "RightNet"]:
            if label in labels and "Ball" in labels:
                goal(label)



# Pygame setup
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2D Soccer Side View")
font = pygame.font.SysFont("Arial", 30)
clock = pygame.time.Clock()

# Box2D world (gravity downwards)
world = b2World(gravity=(0, GRAVITY), doSleep=True)
world.contactListener = MyContactListener()

# Coordinate helpers
def to_screen(pos):
    return int(pos[0] * PPM), int(HEIGHT - pos[1] * PPM)

# Field and ground
ground = world.CreateStaticBody(position=(25, 1), userData = "Ground")
ground.CreatePolygonFixture(box=(25, 1), friction = 0.3)


# Side walls (left/right)
world.CreateStaticBody(position=(0, 15)).CreatePolygonFixture(box=(1, 15))
world.CreateStaticBody(position=(50, 15)).CreatePolygonFixture(box=(1, 15))


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
ball.CreateCircleFixture(radius=0.5, density=1, friction=0.3, restitution=0.8)

# Players
def create_player(x, y, label: str):
    body = world.CreateDynamicBody(position=(x, y), fixedRotation=True,
                                   linearDamping = DAMP_RATE)
    body.CreateCircleFixture(radius=1, density=2, friction=0.1, restitution=0.3,userData = label)
    return body

def goal(net_name):
    global terminated, winner
    if net_name == "LeftNet":
        print("Player 2 wins")
        winner = "Player 2"
    elif net_name == "RightNet":
        print("Player 1 wins")
        winner = "Player 1"

    terminated = True

def player1_kick_check():
    dist = ball.position - player1.position
    if dist.length < 2:
        player1_kick()

def player2_kick_check():
    dist = ball.position - player2.position
    if dist.length < 2:
        player2_kick()

def player1_kick():
    direction = ball.position - player1.position
    if direction.length == 0:
        return
    direction.Normalize()
    ball.ApplyLinearImpulse((direction.x * KICK_STRENGTH, direction.y * KICK_STRENGTH), ball.worldCenter, True)

def player2_kick():
    direction = ball.position - player2.position
    if direction.length == 0:
        return
    direction.Normalize()
    ball.ApplyLinearImpulse((direction.x * KICK_STRENGTH, direction.y * KICK_STRENGTH), ball.worldCenter, True)

player1 = create_player(10, 2, "Player1")
player2 = create_player(40, 2, "Player2")


# Reset positions
def reset_positions():
    ball.position = (25, 10)
    ball.linearVelocity = (0, 0)
    player1.position = (10, 2)
    player1.linearVelocity = (0, 0)
    player2.position = (40, 2)
    player2.linearVelocity = (0, 0)

# if IS_RECORDING:
#     recorder = ScreenRecorder(30)
#     recorder.start_rec()
# Main loop
running = True
while running:
    screen.fill(GREEN)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Input
    keys = pygame.key.get_pressed()
    force = 80

    # Player 1 - WASD
    if keys[pygame.K_a]: player1.ApplyForceToCenter((-force, 0), True)
    if keys[pygame.K_d]: player1.ApplyForceToCenter((force, 0), True)
    if keys[pygame.K_w] and jump_count["Player1"] < 1 and player1_invalidate_jump < 0:
        player1.ApplyLinearImpulse((0, JUMP_FORCE), player1.worldCenter, True)
        jump_count["Player1"] = 1
        player1_invalidate_jump = 3
    if keys[pygame.K_SPACE]: player1_kick_check()

    # Player 2 - Arrows
    if keys[pygame.K_LEFT]: player2.ApplyForceToCenter((-force, 0), True)
    if keys[pygame.K_RIGHT]: player2.ApplyForceToCenter((force, 0), True)
    if keys[pygame.K_UP] and jump_count["Player2"] < 1 and player2_invalidate_jump < 0:
        player2.ApplyLinearImpulse((0, JUMP_FORCE), player2.worldCenter, True)
        jump_count["Player2"] = 1
        player2_invalidate_jump = 3
    if keys[pygame.K_RETURN]: player2_kick_check() #RETURN은 엔터 버튼임.

    # Step physics
    world.Step(TIME_STEP, 10, 10)

    # 변수 처리
    timestep += 1
    player1_invalidate_jump -= 1
    player2_invalidate_jump -= 1

    if timestep >= 1000:
        print("1000 timestep passed. Game ends now")
        terminated = True
    # Draw ground
    pygame.draw.rect(screen, BLACK, pygame.Rect(0, HEIGHT - 2 * PPM, WIDTH, 2 * PPM))

    # Draw goal posts and crossbars
    pygame.draw.rect(screen, YELLOW, (*to_screen((LEFT_GOAL_X, 2 + GOAL_HEIGHT)), 5, int(GOAL_HEIGHT * PPM)))
    pygame.draw.rect(screen, YELLOW, (*to_screen((RIGHT_GOAL_X,2 + GOAL_HEIGHT)), 5, int(GOAL_HEIGHT * PPM)))
    pygame.draw.rect(screen, WHITE, (*to_screen((LEFT_GOAL_X, GOAL_HEIGHT + 2)),  int(CROSSBAR_WIDTH*PPM), int(CROSSBAR_HEIGHT * PPM)))
    pygame.draw.rect(screen, WHITE, (*to_screen((RIGHT_GOAL_X - CROSSBAR_WIDTH, GOAL_HEIGHT + 2)), int(CROSSBAR_WIDTH*PPM), int(CROSSBAR_HEIGHT * PPM)))
    # Draw ball
    pygame.draw.circle(screen, WHITE, to_screen(ball.position), int(0.5 * PPM))

    # Draw players
    pygame.draw.circle(screen, BLUE, to_screen(player1.position), int(1 * PPM))
    pygame.draw.circle(screen, RED, to_screen(player2.position), int(1 * PPM))

    if terminated:
        break

    #버그 수정을 위한 방편2
    if player1.linearVelocity.y > 15.:
        player1.linearVelocity.y = 15
    if player2.linearVelocity.y > 15.:
        player2.linearVelocity.y = 15


    pygame.display.flip()
    clock.tick(TARGET_FPS)

# if IS_RECORDING:
#     recorder.stop_rec()
#     recorder.save_recording("soccer_game_example.mp4")
pygame.time.delay(500)
pygame.quit()