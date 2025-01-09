import pygame
import random
import sys

# Initialize pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Rock Paper Scissors")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Load images
rock_img = pygame.image.load("rock.png")
paper_img = pygame.image.load("paper.png")
scissors_img = pygame.image.load("scissors.png")
choices = ["rock", "paper", "scissors"]

# Scale images
rock_img = pygame.transform.scale(rock_img, (150, 150))
paper_img = pygame.transform.scale(paper_img, (150, 150))
scissors_img = pygame.transform.scale(scissors_img, (150, 150))

# Fonts
font = pygame.font.Font(None, 50)

# Game loop
running = True
user_choice = None
computer_choice = None
result = None

def get_result(user, computer):
    if user == computer:
        return "Draw!"
    elif (user == "rock" and computer == "scissors") or \
         (user == "paper" and computer == "rock") or \
         (user == "scissors" and computer == "paper"):
        return "You Win!"
    else:
        return "You Lose!"

while running:
    screen.fill(WHITE)

    # Display images
    screen.blit(rock_img, (100, 400))
    screen.blit(paper_img, (325, 400))
    screen.blit(scissors_img, (550, 400))

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            if 100 <= x <= 250 and 400 <= y <= 550:  # Rock
                user_choice = "rock"
            elif 325 <= x <= 475 and 400 <= y <= 550:  # Paper
                user_choice = "paper"
            elif 550 <= x <= 700 and 400 <= y <= 550:  # Scissors
                user_choice = "scissors"
            
            if user_choice:
                computer_choice = random.choice(choices)
                result = get_result(user_choice, computer_choice)

    # Display choices
    if user_choice and computer_choice:
        user_text = font.render(f"You chose: {user_choice}", True, BLACK)
        computer_text = font.render(f"Computer chose: {computer_choice}", True, BLACK)
        result_text = font.render(result, True, RED)

        screen.blit(user_text, (50, 50))
        screen.blit(computer_text, (50, 120))
        screen.blit(result_text, (50, 190))

    pygame.display.flip()

# Quit pygame
pygame.quit()
sys.exit()
