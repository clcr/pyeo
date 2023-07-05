import sys
import time
import random

print(f"Python Version: {sys.version_info}")
# print('Python Version: ', sys.version_info)

minimum_duration_seconds = 60
maximum_duration_seconds = minimum_duration_seconds + 30
random_duration_seconds = random.randint(
    minimum_duration_seconds, maximum_duration_seconds
)
print(f"Sleeping for {random_duration_seconds} seconds")
time.sleep(random_duration_seconds)
print(f"Sleep period completed - program ending")
