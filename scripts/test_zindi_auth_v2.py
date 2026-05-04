import os
from dotenv import load_dotenv
from zindi.user import Zindian

load_dotenv()

print("=" * 60)
print("ZINDI AUTH TEST — Real API")
print("=" * 60)

# Real constructor: only username and fixed_password
print("\n[1/4] Initializing Zindian...")
user = Zindian(
    username=os.getenv("ZINDI_USERNAME"),
    fixed_password=os.getenv("ZINDI_PASSWORD")
)
print(f"✅ Logged in as: {os.getenv('ZINDI_USERNAME')}")

# Select a challenge using real API
# select_a_challenge(reward, kind, active, fixed_index)
# fixed_index = None means interactive, pass 0 to pick first active competition
print("\n[2/4] Selecting a challenge...")
user.select_a_challenge(fixed_index=0)
print(f"✅ Challenge: {user.which_challenge}")

# Check remaining submissions
print("\n[3/4] Checking submission budget...")
print(f"✅ Remaining submissions today: {user.remaining_subimissions}")

# Check rank
print("\n[4/4] Checking rank...")
print(f"✅ My rank: {user.my_rank}")

print("\n" + "=" * 60)
print("✅ ALL AUTH CHECKS PASSED")
print("=" * 60)
