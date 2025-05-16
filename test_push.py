from pushbullet import Pushbullet

pb = Pushbullet("o.2AZdZ9oXgTgEQyH3R7ZnwTGzmn5YdpCS")
pb.push_note("✅ Pushbullet Test", "This is a test from your local setup!")
print("Test push sent successfully.")
