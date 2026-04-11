# Akatsuki(Team name)
# SignSense(Project name)
Vihan 9.0 Project

## Real-Time Sign Language to Speech (Arduino + Python)

This project converts live Arduino glove sensor data into:

Gesture -> Text -> Speech

## Sensor Input Format

Arduino sends one CSV line per sample at 115200 baud:

timestamp,f0,f1,f2,f3,f4

- f0-f4: flex sensors