import pickle
import os
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Title Banner
print(Fore.YELLOW + Style.BRIGHT + "\n" + "="*50)
print(Fore.CYAN + Style.BRIGHT + "     💰 GOLD PRICE PREDICTION SYSTEM 💰")
print(Fore.YELLOW + Style.BRIGHT + "="*50)

# Load Model
model_file = "gold_model.pickle"
if not os.path.exists(model_file):
    print(Fore.RED + "❌ Model file not found! Please run 'model_train.py' first.")
    exit()

with open(model_file, "rb") as f:
    model = pickle.load(f)

# Instructions
print(Fore.GREEN + "\n📌 Please enter the following market values:")
print(Fore.MAGENTA + "SPX      → S&P 500 Index value")
print(Fore.MAGENTA + "USO      → Oil Fund price")
print(Fore.MAGENTA + "SLV      → Silver Trust price")
print(Fore.MAGENTA + "EUR/USD  → Euro to US Dollar rate\n")

# Function to take validated float input
def get_float_input(prompt):
    while True:
        try:
            return float(input(Fore.CYAN + prompt + Fore.YELLOW))
        except ValueError:
            print(Fore.RED + "❌ Invalid input! Please enter a numeric value.")

# Get user inputs
spx = get_float_input("Enter SPX value: ")
uso = get_float_input("Enter USO value: ")
slv = get_float_input("Enter SLV value: ")
eurusd = get_float_input("Enter EUR/USD value: ")

# Predict
predicted_price = model.predict([[spx, uso, slv, eurusd]])

# Display Result
print(Fore.YELLOW + "\n" + "="*50)
print(Fore.GREEN + f"💰 Predicted Gold Price: {Fore.WHITE + Style.BRIGHT}{predicted_price[0]:.2f} USD")
print(Fore.YELLOW + "="*50)
print(Fore.BLUE + "📈 Prediction based on given market values\n")
